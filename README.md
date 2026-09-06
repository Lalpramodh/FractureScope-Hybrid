# FractureScope

AI-assisted X-ray fracture screening with a Flask web interface and one YOLOv8 detector.

## Production contract

- Python: `3.11.9`
- Inference: `yolov8_model.pt` only
- Runtime: CPU-only, lazy model loading, image size `512`
- Web server: one Gunicorn worker and one thread
- Database: SQLite, initialized automatically at startup
- Health check: `/health`

## Render

Render can use `render.yaml` directly. The equivalent start command is:

```text
gunicorn app:app --workers 1 --threads 1 --timeout 120 --graceful-timeout 30 --max-requests 20 --max-requests-jitter 5
```

Set `SECRET_KEY` in the Render environment. `YOLO_MODEL_PATH` is optional and defaults to `yolov8_model.pt` in the project root.

## Local verification

```powershell
python -m pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000/health` and confirm the response reports `status: ok` and `model: YOLO-only`. The detector is not loaded until an authenticated request is sent to `/predict`.
