# FractureScope

AI-assisted X-ray fracture screening with a Flask web interface. YOLOv8 performs local fracture-region detection, and Groq provides an optional text chatbot for general questions.

## Production contract

- Python: `3.11.9`
- Inference: ONNX Runtime uses `yolov8_model.onnx`, exported from `yolov8_model.pt`
- Runtime: CPU-only, one ONNX session per worker, dynamic 320px stride-aligned input
- Web server: one Gunicorn worker and one thread
- Database: PostgreSQL in production or SQLite for local development
- Health check: `/health`

## Render

Render can use `render.yaml` directly. The equivalent start command is:

```text
gunicorn app:app --workers 1 --threads 1 --timeout 300
```

Set `SECRET_KEY`, `DATABASE_URL`, and `GROQ_API_KEY` in the Render environment. `GROQ_CHAT_MODEL` is configurable and defaults to `llama-3.1-8b-instant`. `YOLO_ONNX_MODEL_PATH` is optional and defaults to `yolov8_model.onnx`.

## Environment variables

```text
SECRET_KEY=your-secret-key
GROQ_API_KEY=your-groq-api-key
GROQ_CHAT_MODEL=llama-3.1-8b-instant
GROQ_CHAT_MAX_TOKENS=300
DATABASE_URL=your-postgresql-url
```

Never commit real values. When `GROQ_API_KEY` is absent or the API fails, the chatbot displays an unavailable message. YOLO detection does not depend on Groq.

## Local verification

```powershell
python -m pip install -r requirements.txt
python app.py
```

The development-only export uses `requirements-export.txt`, then runs `python export_yolo_onnx.py` from that environment. Render installs only `requirements.txt` and never imports PyTorch or Ultralytics.

Then open `http://localhost:5000/health`. It returns HTTP 200 without authentication and reports `yolo_loaded`, `groq_configured`, and database status without exposing secrets.

Each prediction stores the existing `hybrid_result` JSON column with YOLO region boxes, classes, and confidence values. The result page includes a medical disclaimer.
