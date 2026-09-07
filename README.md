# FractureScope

AI-assisted X-ray fracture screening with a Flask web interface. YOLOv8 performs local detection and Groq Vision provides a possible pattern and explanatory observations for each detected crop.

## Production contract

- Python: `3.11.9`
- Inference: YOLOv8 uses `yolov8_model.pt`; Groq Vision analyzes padded detection crops
- Runtime: CPU-only, one YOLO instance per worker, YOLO inference size `384`, up to five Groq region requests per upload
- Web server: one Gunicorn worker and one thread
- Database: PostgreSQL in production or SQLite for local development
- Health check: `/health`

## Render

Render can use `render.yaml` directly. The equivalent start command is:

```text
gunicorn app:app --workers 1 --threads 1 --timeout 180 --graceful-timeout 30 --max-requests 20 --max-requests-jitter 5
```

Set `SECRET_KEY`, `DATABASE_URL`, and `GROQ_API_KEY` in the Render environment. `GROQ_VISION_MODEL` is configurable and defaults to `meta-llama/llama-4-scout-17b-16e-instruct`. The YOLO model path remains configurable through `YOLO_MODEL_PATH`.

## Environment variables

```text
SECRET_KEY=your-secret-key
GROQ_API_KEY=your-groq-api-key
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
DATABASE_URL=your-postgresql-url
```

Never commit real values. When `GROQ_API_KEY` is absent or the API fails, YOLO results are still saved and displayed with an unavailable-analysis message. Images with no YOLO detections do not call Groq.

## Local verification

```powershell
python -m pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000/health`. It returns HTTP 200 without authentication and reports `yolo_loaded`, `groq_configured`, and database status without exposing secrets.

Each prediction stores the existing `hybrid_result` JSON column. Each region contains `bbox`, `yolo_confidence`, `possible_fracture_type`, `confidence_level`, `description`, `observations`, `limitations`, and `recommendation`. The result page always labels the pattern as possible and includes a medical disclaimer.
