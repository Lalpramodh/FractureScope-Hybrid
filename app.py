import os
# Keep native ML libraries conservative on small Render instances.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
from datetime import datetime

import sqlite3
import torch
from flask import Flask, render_template, request, redirect, session, url_for, flash
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from torchvision import transforms
from ultralytics import YOLO
import timm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-secret-in-render")

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The YOLO model is included in the repository.
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", os.path.join(BASE_DIR, "yolov8_model.pt"))
SWIN_MODEL_PATH = os.getenv("SWIN_MODEL_PATH", os.path.join(BASE_DIR, "model.pth"))

yolo_model = None
swin_model = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
class_names = ["Wrist Fracture", "Humorous Fracture", "Elbow Fracture", "Forearm Fracture"]

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
    """Load YOLO only when a prediction is requested, not when Gunicorn boots."""
    global yolo_model
    if yolo_model is not None:
        return yolo_model
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(
            f"YOLO model not found at {YOLO_MODEL_PATH}. Add yolov8_model.pt to the repository."
        )
    model = YOLO(YOLO_MODEL_PATH)
    yolo_model = model
    return yolo_model

def load_swin_model():
    global swin_model
    if swin_model is not None:
        return swin_model
    if not os.path.exists(SWIN_MODEL_PATH):
        raise FileNotFoundError(
            "model.pth is required for Swin Transformer prediction. "
            "Add it to the project root (preferably through Git LFS) and redeploy."
        )
    model = timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=False,
        num_classes=4,
    )
    state = torch.load(SWIN_MODEL_PATH, map_location=device)
    # Support checkpoints saved as a raw state_dict or wrapped in common checkpoint keys.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    swin_model = model
    return swin_model

@app.route("/health")
def health():
    return {"status": "ok", "model_file_present": os.path.exists(SWIN_MODEL_PATH)}

@app.route("/")
def main():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("SELECT id, username, email, password FROM users WHERE email = ?", (email,))
            user = cur.fetchone()
            cur.close()
            conn.close()
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
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute("SELECT id, username, email, password FROM users WHERE email = ?", (email,))
            if cur.fetchone():
                cur.close()
                conn.close()
                flash("Email already registered. Please login.", "warning")
                return redirect(url_for("login"))
            cur.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, password),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception:
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
            flash("Could not create account. Please check the database configuration.", "danger")
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
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/analysis")
def analysis():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT username, image_path, prediction, timestamp FROM predictions ORDER BY timestamp DESC")
    data = cur.fetchall()
    cur.close()
    conn.close()
    processed_data = []
    for username, image_path, prediction, timestamp in data:
        relative_path = str(image_path).split("static/")[-1]
        processed_data.append({
            "username": username,
            "prediction": prediction,
            "timestamp": timestamp,
            "image_url": url_for("static", filename=relative_path),
        })
    return render_template("analysis.html", data=processed_data)

@app.route("/profile")
def profile():
    if "username" not in session:
        return redirect(url_for("login"))
    username = session["username"]
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT prediction, timestamp FROM predictions WHERE username = ? ORDER BY timestamp DESC",
        (username,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    history = []
    for prediction, timestamp in rows:
        history.append({
            "date": str(timestamp)[:10],
            "time": str(timestamp)[11:19],
            "name": username,
            "result": prediction,
        })
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
        yolo = load_yolo_model()
        yolo_result = yolo(filepath, verbose=False)[0]
        boxes = yolo_result.boxes
        if boxes is None or len(boxes) == 0:
            result = "No fracture detected"
        else:
            xyxy = boxes[0].xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            image = Image.open(filepath).convert("RGB")
            cropped = image.crop((x1, y1, x2, y2))
            img_tensor = transform(cropped).unsqueeze(0).to(device)
            model = load_swin_model()
            with torch.no_grad():
                output = model(img_tensor)
                predicted = torch.argmax(output, dim=1).item()
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence = probabilities[0][predicted].item()
            result = f"{class_names[predicted]} ({confidence * 100:.2f}%)"

        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (username, image_path, prediction) VALUES (?, ?, ?)",
            (session["username"], filepath, result),
        )
        conn.commit()
        cur.close()
        conn.close()
        flash(f"Prediction: {result}", "success")
        return render_template("input.html", prediction=result, image_url=relative_path)
    except FileNotFoundError as exc:
        flash(str(exc), "danger")
        return render_template("input.html", prediction=None, image_url=relative_path), 500
    except Exception as exc:
        app.logger.exception("Prediction failed")
        flash(f"Prediction failed: {exc}", "danger")
        return render_template("input.html", prediction=None, image_url=relative_path), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
