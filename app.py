import os
import sqlite3
import threading
import logging
import re
import gc
import json
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

# Keep native libraries conservative on a small CPU-only Render instance.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from PIL import Image, ImageDraw, UnidentifiedImageError
from flask import Flask, flash, redirect, render_template, request, session, url_for

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:  # PostgreSQL is optional for local SQLite development.
    psycopg2 = None
    RealDictCursor = None
from groq import Groq, RateLimitError
from onnx_yolo import predict as onnx_predict, session_loaded as onnx_session_loaded
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

try:
    import psutil
except ImportError:  # Optional; Render diagnostics still work without it.
    psutil = None

Image.MAX_IMAGE_PIXELS = 50_000_000
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "fracturescope.db"))
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")
GROQ_CHAT_MAX_TOKENS = max(128, int(os.getenv("GROQ_CHAT_MAX_TOKENS", "300")))

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

_inference_lock = threading.Lock()
_groq_client = None


def available_memory_mb():
    """Return cgroup-aware available memory when running under Render/Linux."""
    if os.name == "nt":
        return None
    try:
        memory_limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        memory_current = Path("/sys/fs/cgroup/memory.current").read_text().strip()
        if memory_limit != "max":
            return (int(memory_limit) - int(memory_current)) / 1048576
    except (OSError, ValueError):
        pass
    if psutil is not None:
        return psutil.virtual_memory().available / 1048576
    return None


def reclaim_process_memory():
    gc.collect()
    if os.name != "nt":
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


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


def yolo_predict(filepath):
    preprocessing_started = time.perf_counter()
    image = None
    try:
        with Image.open(filepath) as source:
            source.verify()
        with Image.open(filepath) as source:
            image = source.convert("RGB")
            app.logger.info(
                "Inference image prepared size=%sx%s elapsed=%.2fs",
                image.width,
                image.height,
                time.perf_counter() - preprocessing_started,
            )
        app.logger.info("ONNX INFERENCE START pid=%s available_memory_mb=%s", os.getpid(), available_memory_mb())
        detections = onnx_predict(image)
        app.logger.info("ONNX DETECTIONS count=%s", len(detections))
        return {"detections": detections, "inference_size": image.size}
    except FileNotFoundError:
        app.logger.exception("ONNX MODEL UNAVAILABLE pid=%s", os.getpid())
        raise
    except Exception:
        app.logger.exception("ONNX INFERENCE FAILED pid=%s", os.getpid())
        raise
    finally:
        if image is not None:
            image.close()
        gc.collect()


def _annotate_image(filepath, detections):
    annotated_name = f"annotated_{Path(filepath).name}"
    annotated_path = UPLOAD_FOLDER / annotated_name
    with Image.open(filepath) as source:
        image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for index, detection in enumerate(detections, start=1):
            x1, y1, x2, y2 = detection["bbox"]
            label = f"#{index} YOLO {detection['yolo_confidence'] * 100:.1f}%"
            draw.rectangle((x1, y1, x2, y2), outline="#45c69c", width=max(3, image.width // 350))
            text_box = draw.textbbox((x1, y1), label)
            text_height = text_box[3] - text_box[1]
            text_y = max(0, y1 - text_height - 8)
            draw.rectangle((x1, text_y, x1 + (text_box[2] - text_box[0]) + 10, y1), fill="#10242b")
            draw.text((x1 + 5, text_y + 3), label, fill="#82e3c0")
        image.save(annotated_path, format="JPEG", quality=92)
    return annotated_path


def _groq_client_instance():
    global _groq_client
    if _groq_client is None and os.getenv("GROQ_API_KEY"):
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


def chat_with_groq(messages):
    client = _groq_client_instance()
    if client is None:
        return "Chat is unavailable because GROQ_API_KEY is not configured."
    try:
        response = client.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            temperature=0,
            max_tokens=GROQ_CHAT_MAX_TOKENS,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    except RateLimitError as exc:
        app.logger.warning("GROQ CHAT RATE LIMIT message=%s", str(exc)[:240])
        return "Chat is temporarily unavailable because the Groq request limit was reached."
    except Exception:
        app.logger.exception("GROQ CHAT FAILED")
        return "Chat is temporarily unavailable. Please try again later."


def _run_yolo_prediction(filepath):
    started = time.perf_counter()
    app.logger.info("YOLO prediction started")
    results = yolo_predict(filepath)
    if not results["detections"]:
        app.logger.info("YOLO detections=%s", 0)
        return {"detections": [], "summary": "No fracture region detected by the YOLO model."}, None
    original_image = None
    try:
        with Image.open(filepath) as original:
            original_image = original.convert("RGB")
        original_width, original_height = original_image.size
        inference_width, inference_height = results["inference_size"]
        detections = []
        for yolo_detection in results["detections"]:
            raw_box = yolo_detection["raw_bbox"]
            scale_x = original_width / inference_width
            scale_y = original_height / inference_height
            bbox = [round(raw_box[0] * scale_x), round(raw_box[1] * scale_y), round(raw_box[2] * scale_x), round(raw_box[3] * scale_y)]
            detections.append({
                "bbox": bbox,
                "yolo_class": yolo_detection["yolo_class"],
                "yolo_confidence": yolo_detection["yolo_confidence"],
            })
        app.logger.info("YOLO detections=%s", len(detections))
        annotated_path = _annotate_image(filepath, detections)
        result = {"detections": detections, "summary": f"{len(detections)} fracture region(s) detected"}
        app.logger.info("YOLO prediction completed in %.2fs", time.perf_counter() - started)
        return result, annotated_path
    finally:
        del results
        if original_image is not None:
            original_image.close()
        reclaim_process_memory()


def run_yolo_prediction(filepath):
    """Run one complete YOLO operation without concurrent inference."""
    with _inference_lock:
        return _run_yolo_prediction(filepath)


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


@app.route("/health")
def health():
    return {
        "status": "healthy",
        "yolo_loaded": onnx_session_loaded(),
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "database_configured": using_postgres() or DB_PATH.is_file(),
        "available_memory_mb": available_memory_mb(),
        "process_id": os.getpid(),
    }, 200


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    incoming = payload.get("messages", [])
    if not isinstance(incoming, list):
        return {"reply": "Please send a valid chat message."}, 400
    messages = []
    for item in incoming[-10:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        content = str(item.get("content", "")).strip()
        if content:
            messages.append({"role": item["role"], "content": content[:1200]})
    if not messages or messages[-1]["role"] != "user":
        return {"reply": "Please enter a question first."}, 400
    system_message = {
        "role": "system",
        "content": (
            "You are FractureScope's helpful health-information assistant. Be concise and clear. "
            "You may explain X-rays, fracture terminology, the app workflow, and general safety guidance. "
            "Do not diagnose, confirm a fracture, interpret an individual image, infer patient details, "
            "or replace a radiologist or healthcare professional. For urgent symptoms, advise professional care."
        ),
    }
    started = time.perf_counter()
    reply = chat_with_groq([system_message, *messages])
    app.logger.info("GROQ CHAT COMPLETE elapsed=%.2fs", time.perf_counter() - started)
    return {"reply": reply}, 200


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
        hybrid_result, annotated_path = run_yolo_prediction(str(filepath))
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
        flash(f"AI analysis: {result}", "success")
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
    except FileNotFoundError:
        flash("Prediction is temporarily unavailable because the ONNX model is not installed.", "danger")
        app.logger.exception("Prediction failed because the ONNX model is missing")
        return redirect(url_for("input"))
    except Exception:
        app.logger.exception("ONNX prediction failed")
        flash("Prediction could not be completed. Confirm the image and try again.", "danger")
        return redirect(url_for("input"))
    finally:
        if filepath is not None and not keep_upload:
            filepath.unlink(missing_ok=True)

try:
    init_db()
    app.logger.info(
        "WORKER READY pid=%s onnx_session_loaded=%s groq_configured=%s",
        os.getpid(),
        onnx_session_loaded(),
        bool(os.getenv("GROQ_API_KEY")),
    )
except Exception:
    app.logger.exception("WORKER STARTUP FAILED pid=%s", os.getpid())
    raise

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
