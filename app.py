import os
import sqlite3
import threading
from pathlib import Path
from secrets import token_hex
from uuid import uuid4

# Keep native libraries conservative on a small CPU-only Render instance.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import torch
from PIL import Image, UnidentifiedImageError
from flask import Flask, flash, redirect, render_template, request, session, url_for
from ultralytics import YOLO
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", BASE_DIR / "yolov8_model.pt"))
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "fracturescope.db"))
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY") or token_hex(32)
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    UPLOAD_FOLDER=str(UPLOAD_FOLDER),
)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_model = None
_model_lock = threading.Lock()


def get_db():
    connection = sqlite3.connect(str(DB_PATH), timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                image_path TEXT NOT NULL,
                prediction TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)


def load_yolo_model():
    """Load the single detector once, only when the first prediction needs it."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"YOLO model not found at {MODEL_PATH}")
        detector = YOLO(str(MODEL_PATH))
        detector.to("cpu")
        detector.overrides.update({
            "device": "cpu",
            "imgsz": 512,
            "half": False,
            "verbose": False,
        })
        _model = detector
        return _model


def yolo_predict(filepath):
    detector = load_yolo_model()
    with torch.inference_mode():
        results = detector.predict(
            source=filepath,
            device="cpu",
            imgsz=512,
            conf=0.25,
            iou=0.45,
            half=False,
            max_det=20,
            verbose=False,
        )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return "No fracture detected"

    result = results[0]
    boxes = result.boxes
    best_index = int(boxes.conf.argmax().item())
    class_id = int(boxes.cls[best_index].item())
    confidence = float(boxes.conf[best_index].item())
    names = result.names or getattr(detector, "names", {})
    label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
    return f"{label} ({confidence * 100:.2f}%)"


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
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc
    filename = f"{uuid4().hex}_{original_name}"
    filepath = UPLOAD_FOLDER / filename
    upload.save(filepath)
    return filepath, f"uploads/{filename}"


init_db()

@app.route("/health")
def health():
    return {
        "status": "ok",
        "model": "YOLO-only",
        "yolo_model_present": MODEL_PATH.is_file(),
        "model_loaded": _model is not None,
        "database": DB_PATH.name,
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
                user = connection.execute(
                    "SELECT id, username, email, password FROM users WHERE email = ?", (email,)
                ).fetchone()
        except Exception:
            flash("Database connection failed. Please try again later.", "danger")
            return redirect(url_for("login"))
        if user and check_password_hash(user["password"], password):
            session["username"] = user["username"]
            session["email"] = user["email"]
            return redirect(url_for("home"))
        flash("Invalid credentials. Please try again or register.", "danger")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        raw_password = request.form.get("password", "")
        if not username or not email or len(raw_password) < 8:
            flash("Provide a username, email, and password of at least 8 characters.", "warning")
            return redirect(url_for("register"))
        try:
            with get_db() as connection:
                if connection.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone():
                    flash("Email already registered. Please login.", "warning")
                    return redirect(url_for("login"))
                connection.execute(
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
    if "username" in session:
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
    if "username" not in session: return redirect(url_for("login"))
    username = session["username"]
    with get_db() as connection:
        rows = connection.execute(
            "SELECT prediction, timestamp FROM predictions WHERE username = ? ORDER BY timestamp DESC",
            (username,),
        ).fetchall()
    history = [{"date": str(r["timestamp"])[:10], "time": str(r["timestamp"])[11:19], "name": username, "result": r["prediction"]} for r in rows]
    return render_template("profile.html", history=history)

@app.route("/input", methods=["GET", "POST"])
def input():
    return render_template("input.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "username" not in session:
        flash("Please log in to make predictions.", "warning")
        return redirect(url_for("login"))
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        flash("Choose an image before running a screening.", "danger")
        return redirect(url_for("input"))

    try:
        filepath, relative_path = save_upload(upload)
        result = yolo_predict(str(filepath))
        with get_db() as connection:
            connection.execute(
                "INSERT INTO predictions (username, image_path, prediction) VALUES (?, ?, ?)",
                (session["username"], str(filepath), result),
            )
        flash(f"Prediction: {result}", "success")
        return render_template("input.html", prediction=result, image_url=relative_path)
    except ValueError as exc:
        flash(str(exc), "warning")
        return redirect(url_for("input"))
    except Exception as exc:
        app.logger.exception("YOLO prediction failed")
        flash("Prediction could not be completed. Confirm the image and try again.", "danger")
        return redirect(url_for("input"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
