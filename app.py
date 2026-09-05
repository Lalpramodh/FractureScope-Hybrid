import os
from datetime import datetime
import sqlite3

# Keep native ML libraries conservative on small Render instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from flask import Flask, render_template, request, redirect, session, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-secret-in-render")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# FractureScope now uses YOLO as the ONLY prediction model.
# The previous Swin Transformer/model.pth is intentionally ignored.
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", os.path.join(BASE_DIR, "yolov8_model.pt"))
yolo_model = None

DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "fracturescope.db"))

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
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
    conn.commit()
    conn.close()

init_db()

def load_yolo_model():
    global yolo_model
    if yolo_model is not None:
        return yolo_model
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_MODEL_PATH}. "
            "Make sure yolov8_model.pt is present in the repository."
        )
    model = YOLO(YOLO_MODEL_PATH)
    # CPU-only inference and smaller images are friendlier to Render Free.
    model.overrides.update({
        "device": "cpu",
        "imgsz": 512,
        "half": False,
        "verbose": False,
    })
    # Ultralytics may fuse Conv+BatchNorm during predictor setup. On small
    # CPU instances that fusion can spike memory and cause exit code 139.
    # Keep the loaded model unfused for safer inference.
    if getattr(model, "model", None) is not None and hasattr(model.model, "fuse"):
        model.model.fuse = lambda verbose=False: model.model
    yolo_model = model
    return yolo_model

def yolo_predict(filepath):
    model = load_yolo_model()
    results = model.predict(
        source=filepath,
        device="cpu",
        imgsz=512,
        conf=0.25,
        iou=0.45,
        half=False,
        verbose=False,
        max_det=20,
    )
    if not results:
        return "No fracture detected"

    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return "No fracture detected"

    names = result.names or getattr(model, "names", {})
    best_index = int(boxes.conf.argmax().item()) if boxes.conf is not None else 0
    class_id = int(boxes.cls[best_index].item()) if boxes.cls is not None else 0
    confidence = float(boxes.conf[best_index].item()) if boxes.conf is not None else 0.0
    label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
    return f"{label} ({confidence * 100:.2f}%)"

@app.route("/health")
def health():
    return {"status": "ok", "yolo_model_present": os.path.exists(YOLO_MODEL_PATH), "model": "YOLO-only"}

@app.route("/")
def main():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            conn = get_db(); cur = conn.cursor()
            cur.execute("SELECT id, username, email, password FROM users WHERE email = ?", (email,))
            user = cur.fetchone(); cur.close(); conn.close()
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
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])
        conn = None
        try:
            conn = get_db(); cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE email = ?", (email,))
            if cur.fetchone():
                cur.close(); conn.close()
                flash("Email already registered. Please login.", "warning")
                return redirect(url_for("login"))
            cur.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
            conn.commit(); cur.close(); conn.close()
        except Exception:
            if conn:
                try: conn.rollback(); conn.close()
                except Exception: pass
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
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT username, image_path, prediction, timestamp FROM predictions ORDER BY timestamp DESC")
    data = cur.fetchall(); cur.close(); conn.close()
    processed_data = []
    for row in data:
        relative_path = str(row["image_path"]).split("static/")[-1]
        processed_data.append({"username": row["username"], "prediction": row["prediction"], "timestamp": row["timestamp"], "image_url": url_for("static", filename=relative_path)})
    return render_template("analysis.html", data=processed_data)

@app.route("/profile")
def profile():
    if "username" not in session: return redirect(url_for("login"))
    username = session["username"]
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT prediction, timestamp FROM predictions WHERE username = ? ORDER BY timestamp DESC", (username,))
    rows = cur.fetchall(); cur.close(); conn.close()
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
    if "image" not in request.files or request.files["image"].filename == "":
        flash("No image uploaded.", "danger")
        return redirect(url_for("input"))

    file = request.files["image"]
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    relative_path = os.path.join("uploads", filename).replace("\\", "/")

    try:
        # YOLO is the sole prediction model; model.pth/Swin is not used.
        result = yolo_predict(filepath)
        conn = get_db(); cur = conn.cursor()
        cur.execute("INSERT INTO predictions (username, image_path, prediction) VALUES (?, ?, ?)", (session["username"], filepath, result))
        conn.commit(); cur.close(); conn.close()
        flash(f"Prediction: {result}", "success")
        return render_template("input.html", prediction=result, image_url=relative_path)
    except Exception as exc:
        app.logger.exception("YOLO prediction failed")
        flash(f"Prediction failed: {exc}", "danger")
        return render_template("input.html", prediction=None, image_url=relative_path), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
