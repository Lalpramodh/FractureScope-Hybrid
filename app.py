import os
import sqlite3
import threading
import logging
import re
import gc
import json
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

# Keep native libraries conservative on a small CPU-only Render instance.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import torch
from PIL import Image, ImageDraw, UnidentifiedImageError
from flask import Flask, flash, redirect, render_template, request, session, url_for

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:  # PostgreSQL is optional for local SQLite development.
    psycopg2 = None
    RealDictCursor = None
from ultralytics import YOLO
import timm
from timm.data import create_transform
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

try:
    import psutil
except ImportError:  # Optional; Render diagnostics still work without it.
    psutil = None

Image.MAX_IMAGE_PIXELS = 50_000_000
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Torch inter-op threads can only be configured before parallel work starts.
    pass

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", BASE_DIR / "yolov8_model.pt"))
SWIN_MODEL_PATH = Path(os.getenv("SWIN_MODEL_PATH", BASE_DIR / "model.pth"))
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "fracturescope.db"))
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("SECRET_KEY must be configured")
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    UPLOAD_FOLDER=str(UPLOAD_FOLDER),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "true").lower() == "true",
)
app.logger.setLevel(logging.INFO)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
if not os.getenv("DATABASE_URL"):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_model = None
_model_lock = threading.Lock()
_swin_model = None
_swin_transform = None
_swin_lock = threading.Lock()
INFERENCE_SIZE = 384
MAX_INFERENCE_DIMENSION = 2048
SWIN_INPUT_SIZE = 224
SWIN_CROP_PADDING = float(os.getenv("SWIN_CROP_PADDING", "0.08"))
SWIN_CLASS_NAMES = [name.strip() for name in os.getenv("SWIN_CLASS_NAMES", "").split(",") if name.strip()]
SWIN_EXCLUSIVE_MEMORY = os.getenv("SWIN_EXCLUSIVE_MEMORY", "true").lower() == "true"


def reclaim_process_memory():
    gc.collect()
    if os.name != "nt":
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


def release_yolo_for_swin():
    global _model
    if _model is not None:
        app.logger.info("Releasing YOLO memory for Swin classification")
        _model = None
        reclaim_process_memory()


def release_swin_after_prediction():
    global _swin_model, _swin_transform
    if _swin_model is not None:
        app.logger.info("Releasing Swin memory after classification")
        _swin_model = None
        _swin_transform = None
        reclaim_process_memory()


def using_postgres():
    return bool(os.getenv("DATABASE_URL"))


@contextmanager
def get_db():
    connection = None
    try:
        if using_postgres():
            if psycopg2 is None:
                raise RuntimeError("psycopg2 is required when DATABASE_URL is configured")
            database_url = os.environ["DATABASE_URL"]
            if database_url.startswith("postgres://"):
                database_url = "postgresql://" + database_url[len("postgres://"):]
            connection = psycopg2.connect(database_url, cursor_factory=RealDictCursor)
        else:
            connection = sqlite3.connect(str(DB_PATH), timeout=30)
            connection.row_factory = sqlite3.Row
        yield connection
        connection.commit()
    except Exception:
        if connection is not None:
            connection.rollback()
        raise
    finally:
        if connection is not None:
            connection.close()


def execute(connection, query, parameters=()):
    if using_postgres():
        query = query.replace("?", "%s")
        cursor = connection.cursor()
        cursor.execute(query, parameters)
        return cursor
    return connection.execute(query, parameters)


def init_db():
    with get_db() as connection:
        if using_postgres():
            connection.cursor().execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    image_path TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    hybrid_result TEXT,
                    annotated_image_path TEXT,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                ALTER TABLE predictions ADD COLUMN IF NOT EXISTS hybrid_result TEXT;
                ALTER TABLE predictions ADD COLUMN IF NOT EXISTS annotated_image_path TEXT;
            """)
        else:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    username TEXT,
                    image_path TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    hybrid_result TEXT,
                    annotated_image_path TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(predictions)").fetchall()}
            if "user_id" not in columns:
                connection.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER")
            if "username" not in columns:
                connection.execute("ALTER TABLE predictions ADD COLUMN username TEXT")
            if "hybrid_result" not in columns:
                connection.execute("ALTER TABLE predictions ADD COLUMN hybrid_result TEXT")
            if "annotated_image_path" not in columns:
                connection.execute("ALTER TABLE predictions ADD COLUMN annotated_image_path TEXT")
            connection.execute("""
                UPDATE predictions SET user_id = (
                    SELECT id FROM users WHERE users.username = predictions.username
                ) WHERE user_id IS NULL AND username IS NOT NULL
            """)
    app.logger.info("Database initialized using %s", "PostgreSQL" if using_postgres() else "SQLite")


def load_yolo_model():
    """Load and configure one CPU detector for the lifetime of this worker."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"YOLO model not found at {MODEL_PATH}")
        model_started = time.perf_counter()
        app.logger.info(
            "MODEL LOAD START pid=%s path=%s",
            os.getpid(),
            MODEL_PATH,
        )
        detector = YOLO(str(MODEL_PATH))
        detector.to("cpu")
        detector.overrides.update({
            "device": "cpu",
            "imgsz": INFERENCE_SIZE,
            "half": False,
            "verbose": False,
        })
        detector.fuse()
        warmup_started = time.perf_counter()
        warmup_image = Image.new("RGB", (INFERENCE_SIZE, INFERENCE_SIZE))
        try:
            with torch.inference_mode():
                warmup_results = detector.predict(
                    source=warmup_image,
                    device="cpu",
                    imgsz=INFERENCE_SIZE,
                    batch=1,
                    half=False,
                    max_det=20,
                    verbose=False,
                    save=False,
                    show=False,
                    stream=False,
                )
        finally:
            warmup_image.close()
            if "warmup_results" in locals():
                del warmup_results
            gc.collect()
        torch.set_num_threads(1)
        _model = detector
        app.logger.info(
            "MODEL LOAD COMPLETE pid=%s elapsed=%.2fs warmup=%.2fs torch_threads=%s",
            os.getpid(),
            time.perf_counter() - model_started,
            time.perf_counter() - warmup_started,
            torch.get_num_threads(),
        )
        return _model


def yolo_predict(filepath):
    prediction_started = time.perf_counter()
    app.logger.info("INFERENCE REQUEST START pid=%s", os.getpid())
    detector = load_yolo_model()
    preprocessing_started = time.perf_counter()
    inference_path = prepare_inference_image(filepath)
    app.logger.info(
        "Image preprocessing completed in %.2f seconds",
        time.perf_counter() - preprocessing_started,
    )
    results = None
    try:
        inference_started = time.perf_counter()
        app.logger.info("INFERENCE START pid=%s", os.getpid())
        with torch.inference_mode():
            results = detector.predict(
                source=str(inference_path),
                device="cpu",
                imgsz=INFERENCE_SIZE,
                batch=1,
                conf=0.25,
                iou=0.45,
                half=False,
                max_det=20,
                verbose=False,
                save=False,
                show=False,
                stream=False,
            )
        app.logger.info(
            "INFERENCE COMPLETE pid=%s elapsed=%.2fs",
            os.getpid(),
            time.perf_counter() - inference_started,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return {"detections": [], "inference_size": None}
        result = results[0]
        names = result.names or getattr(detector, "names", {})
        detections = []
        for index in range(len(result.boxes)):
            detections.append({
                "raw_bbox": [float(value) for value in result.boxes.xyxy[index].tolist()],
                "yolo_class_id": int(result.boxes.cls[index].item()),
                "yolo_class": names.get(int(result.boxes.cls[index].item()), str(int(result.boxes.cls[index].item()))) if isinstance(names, dict) else str(int(result.boxes.cls[index].item())),
                "yolo_confidence": float(result.boxes.conf[index].item()),
            })
        with Image.open(inference_path) as inference_image:
            inference_size = inference_image.size
        return {"detections": detections, "inference_size": inference_size}
    finally:
        if results:
            del results
        if inference_path != Path(filepath):
            inference_path.unlink(missing_ok=True)
        gc.collect()


def load_swin_model():
    """Load the timm Swin-Base checkpoint once for this worker."""
    global _swin_model, _swin_transform
    if _swin_model is not None:
        return _swin_model, _swin_transform
    with _swin_lock:
        if _swin_model is not None:
            return _swin_model, _swin_transform
        if not SWIN_MODEL_PATH.is_file():
            raise FileNotFoundError(f"Swin model not found at {SWIN_MODEL_PATH}")
        started = time.perf_counter()
        app.logger.info("Swin lazy loading started path=%s", SWIN_MODEL_PATH)
        if psutil is not None:
            app.logger.info("Memory before Swin load: %.1f MB", psutil.Process().memory_info().rss / 1048576)
        checkpoint = None
        model = None
        try:
            checkpoint = torch.load(SWIN_MODEL_PATH, map_location="cpu", weights_only=True, mmap=True)
            if not isinstance(checkpoint, dict) or "head.fc.weight" not in checkpoint:
                raise ValueError("Swin checkpoint is not a supported timm state dict")
            class_count = int(checkpoint["head.fc.weight"].shape[0])
            if len(SWIN_CLASS_NAMES) not in (0, class_count):
                raise ValueError(f"SWIN_CLASS_NAMES must contain exactly {class_count} labels")
            model = timm.create_model(
                "swin_base_patch4_window7_224",
                pretrained=False,
                num_classes=class_count,
            )
            model = model.half()
            model_state = model.state_dict()
            with torch.no_grad():
                for key, value in checkpoint.items():
                    model_state[key].copy_(value)
            del model_state
            model.to("cpu").eval()
            _swin_transform = create_transform(
                input_size=(3, SWIN_INPUT_SIZE, SWIN_INPUT_SIZE),
                is_training=False,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                interpolation="bicubic",
            )
            _swin_model = model
            labels = SWIN_CLASS_NAMES or [f"Checkpoint class {index}" for index in range(class_count)]
            if psutil is not None:
                app.logger.info("Memory after Swin load: %.1f MB", psutil.Process().memory_info().rss / 1048576)
            app.logger.info(
                "Swin model loaded in %.2fs architecture=swin_base_patch4_window7_224 classes=%s labels=%s",
                time.perf_counter() - started,
                class_count,
                labels,
            )
            return _swin_model, _swin_transform
        finally:
            del checkpoint
            gc.collect()


def classify_fracture(crop):
    started = time.perf_counter()
    app.logger.info("Swin classification started")
    model, transform = load_swin_model()
    tensor = transform(crop.convert("RGB")).unsqueeze(0).half()
    try:
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
            class_id = int(probabilities.argmax().item())
            confidence = float(probabilities[class_id].item())
        labels = SWIN_CLASS_NAMES or [f"Checkpoint class {index}" for index in range(len(probabilities))]
        app.logger.info("Swin classification completed in %.2fs", time.perf_counter() - started)
        return labels[class_id], confidence
    finally:
        del tensor
        gc.collect()


def _padded_box(box, width, height):
    x1, y1, x2, y2 = box
    padding_x = (x2 - x1) * SWIN_CROP_PADDING
    padding_y = (y2 - y1) * SWIN_CROP_PADDING
    return (
        max(0, int(x1 - padding_x)),
        max(0, int(y1 - padding_y)),
        min(width, int(x2 + padding_x)),
        min(height, int(y2 + padding_y)),
    )


def _annotate_image(filepath, detections):
    annotated_name = f"annotated_{Path(filepath).name}"
    annotated_path = UPLOAD_FOLDER / annotated_name
    with Image.open(filepath) as source:
        image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for index, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection["bbox"]
            label = f"#{index} YOLO {detection['yolo_confidence'] * 100:.1f}% | {detection['swin_class']} {detection['swin_confidence'] * 100:.1f}%" if detection["swin_confidence"] is not None else f"#{index} YOLO {detection['yolo_confidence'] * 100:.1f}% | {detection['swin_class']}"
            draw.rectangle((x1, y1, x2, y2), outline="#45c69c", width=max(3, image.width // 350))
            text_box = draw.textbbox((x1, y1), label)
            text_height = text_box[3] - text_box[1]
            text_y = max(0, y1 - text_height - 8)
            draw.rectangle((x1, text_y, x1 + (text_box[2] - text_box[0]) + 10, y1), fill="#10242b")
            draw.text((x1 + 5, text_y + 3), label, fill="#82e3c0")
        image.save(annotated_path, format="JPEG", quality=92)
    return annotated_path


def run_hybrid_prediction(filepath):
    started = time.perf_counter()
    app.logger.info("Hybrid prediction started")
    results = yolo_predict(filepath)
    if not results["detections"]:
        return {"detections": [], "summary": "No fracture detected"}, None
    if SWIN_EXCLUSIVE_MEMORY:
        release_yolo_for_swin()
    with Image.open(filepath) as original:
        original_image = original.convert("RGB")
        original_width, original_height = original_image.size
    try:
        inference_width, inference_height = results["inference_size"]
        detections = []
        for index, yolo_detection in enumerate(results["detections"]):
            raw_box = yolo_detection["raw_bbox"]
            scale_x = original_width / inference_width
            scale_y = original_height / inference_height
            bbox = [round(raw_box[0] * scale_x), round(raw_box[1] * scale_y), round(raw_box[2] * scale_x), round(raw_box[3] * scale_y)]
            crop_box = _padded_box(bbox, original_width, original_height)
            crop = original_image.crop(crop_box)
            try:
                swin_class, swin_confidence = classify_fracture(crop)
            except Exception:
                app.logger.exception("Swin classification failed for region %s", index + 1)
                swin_class, swin_confidence = "Classification unavailable", None
            finally:
                crop.close()
            detections.append({
                "bbox": bbox,
                "yolo_class": yolo_detection["yolo_class"],
                "yolo_confidence": yolo_detection["yolo_confidence"],
                "swin_class": swin_class,
                "swin_confidence": swin_confidence,
            })
        annotated_path = _annotate_image(filepath, detections)
        result = {"detections": detections, "summary": f"{len(detections)} fracture region(s) detected"}
        app.logger.info("Hybrid prediction completed in %.2fs", time.perf_counter() - started)
        return result, annotated_path
    finally:
        del results
        if SWIN_EXCLUSIVE_MEMORY:
            release_swin_after_prediction()
            try:
                load_yolo_model()
            except Exception:
                app.logger.exception("YOLO restoration failed after Swin classification")
        reclaim_process_memory()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(upload):
    original_name = secure_filename(upload.filename or "")
    if not original_name or not allowed_file(original_name):
        raise ValueError("Upload a JPG, JPEG, or PNG image.")
    try:
        image = Image.open(upload.stream)
        image.verify()
        upload.stream.seek(0)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc
    filename = f"{uuid4().hex}_{original_name}"
    filepath = UPLOAD_FOLDER / filename
    upload.save(filepath)
    return filepath, f"uploads/{filename}"


def prepare_inference_image(filepath):
    """Bound inference memory for unusually large uploads while retaining the original."""
    source = Path(filepath)
    with Image.open(source) as image:
        image.verify()
    with Image.open(source) as image:
        app.logger.info("Inference image dimensions: %sx%s", image.width, image.height)
        if max(image.size) <= MAX_INFERENCE_DIMENSION:
            return source
        image.thumbnail((MAX_INFERENCE_DIMENSION, MAX_INFERENCE_DIMENSION), Image.Resampling.LANCZOS)
        temporary = tempfile.NamedTemporaryFile(suffix=".jpg", dir=UPLOAD_FOLDER, delete=False)
        temporary_path = Path(temporary.name)
        temporary.close()
        image.convert("RGB").save(temporary_path, format="JPEG", quality=95)
        return temporary_path


@app.route("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "database_configured": using_postgres() or DB_PATH.is_file(),
    }, 200


@app.errorhandler(413)
def request_entity_too_large(_error):
    flash("Image is too large. Upload a file smaller than 16 MB.", "warning")
    return redirect(url_for("input"))

@app.route("/")
def main():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        try:
            with get_db() as connection:
                user = execute(connection,
                    "SELECT id, username, email, password FROM users WHERE email = ?", (email,)
                ).fetchone()
        except Exception:
            flash("Database connection failed. Please try again later.", "danger")
            return redirect(url_for("login"))
        if user and check_password_hash(user["password"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("home"))
        app.logger.info("Authentication failed for supplied email")
        flash("Invalid credentials. Please try again or register.", "danger")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        raw_password = request.form.get("password", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{3,40}", username) or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) or len(raw_password) < 8:
            flash("Provide a username, email, and password of at least 8 characters.", "warning")
            return redirect(url_for("register"))
        try:
            with get_db() as connection:
                if execute(connection, "SELECT id FROM users WHERE email = ? OR username = ?", (email, username)).fetchone():
                    flash("That username or email is already registered. Please login.", "warning")
                    return redirect(url_for("login"))
                execute(connection,
                    "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                    (username, email, generate_password_hash(raw_password)),
                )
        except sqlite3.IntegrityError:
            flash("Email already registered. Please login.", "warning")
            return redirect(url_for("login"))
        except Exception:
            app.logger.exception("Registration failed")
            flash("Could not create account. Please try again.", "danger")
            return redirect(url_for("register"))
        flash("Account created. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/forgot-password")
def forgot_password():
    return render_template("forgot.html")

@app.route("/home")
def home():
    if "user_id" in session:
        return render_template("home.html", username=session["username"])
    return redirect(url_for("login"))

@app.route("/logout")
def logout():
    session.clear(); flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/profile")
def profile():
    if "user_id" not in session: return redirect(url_for("login"))
    username = session["username"]
    with get_db() as connection:
        rows = execute(connection,
            "SELECT prediction, hybrid_result, timestamp FROM predictions WHERE user_id = ? ORDER BY timestamp DESC",
            (session["user_id"],),
        ).fetchall()
    history = []
    for row in rows:
        hybrid = None
        if row["hybrid_result"]:
            try:
                hybrid = json.loads(row["hybrid_result"])
            except (TypeError, json.JSONDecodeError):
                app.logger.warning("Ignoring malformed hybrid history record")
        history.append({
            "date": str(row["timestamp"])[:10],
            "time": str(row["timestamp"])[11:19],
            "name": username,
            "result": row["prediction"],
            "hybrid": hybrid,
        })
    return render_template("profile.html", history=history)

@app.route("/input", methods=["GET", "POST"])
def input():
    return render_template("input.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        flash("Please log in to make predictions.", "warning")
        return redirect(url_for("login"))
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        flash("Choose an image before running a screening.", "danger")
        return redirect(url_for("input"))

    request_started = time.perf_counter()
    app.logger.info("PREDICTION REQUEST ACCEPTED pid=%s", os.getpid())
    filepath = None
    keep_upload = False
    try:
        filepath, relative_path = save_upload(upload)
        app.logger.info("UPLOAD VALIDATED pid=%s", os.getpid())
        hybrid_result, annotated_path = run_hybrid_prediction(str(filepath))
        result = hybrid_result["summary"]
        with get_db() as connection:
            if using_postgres():
                execute(connection,
                    "INSERT INTO predictions (user_id, image_path, prediction, hybrid_result, annotated_image_path) VALUES (?, ?, ?, ?, ?)",
                    (session["user_id"], str(filepath), result, json.dumps(hybrid_result), str(annotated_path) if annotated_path else None),
                )
            else:
                execute(connection,
                    "INSERT INTO predictions (user_id, username, image_path, prediction, hybrid_result, annotated_image_path) VALUES (?, ?, ?, ?, ?, ?)",
                    (session["user_id"], session["username"], str(filepath), result, json.dumps(hybrid_result), str(annotated_path) if annotated_path else None),
                )
        app.logger.info("DATABASE SAVE COMPLETE pid=%s", os.getpid())
        flash(f"Hybrid analysis: {result}", "success")
        keep_upload = True
        app.logger.info(
            "Prediction request completed in %.2f seconds",
            time.perf_counter() - request_started,
        )
        display_path = f"uploads/{Path(annotated_path).name}" if annotated_path else relative_path
        return render_template("input.html", prediction=result, image_url=display_path, hybrid_result=hybrid_result)
    except ValueError as exc:
        flash(str(exc), "warning")
        return redirect(url_for("input"))
    except Exception:
        app.logger.exception("YOLO prediction failed")
        flash("Prediction could not be completed. Confirm the image and try again.", "danger")
        return redirect(url_for("input"))
    finally:
        if filepath is not None and not keep_upload:
            filepath.unlink(missing_ok=True)

try:
    init_db()
    load_yolo_model()
    app.logger.info(
        "WORKER READY pid=%s yolo_loaded=%s swin_loaded=%s",
        os.getpid(),
        _model is not None,
        _swin_model is not None,
    )
except Exception:
    app.logger.exception("WORKER STARTUP FAILED pid=%s", os.getpid())
    raise

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
