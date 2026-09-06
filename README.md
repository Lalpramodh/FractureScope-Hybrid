# FractureScope

AI-assisted X-ray fracture screening with a Flask web interface and a hybrid YOLOv8 detector plus Swin Transformer classifier.

## Production contract

- Python: `3.11.9`
- Inference: YOLOv8 uses `yolov8_model.pt`; Swin uses the raw state dict in `model.pth`
- Runtime: CPU-only, one YOLO instance per worker, YOLO inference size `384`, Swin input size `224`; Render uses exclusive model residency because the worker has approximately 512 MiB RAM
- Web server: one Gunicorn worker and one thread
- Database: PostgreSQL in production or SQLite for local development
- Health check: `/health`

## Render

Render can use `render.yaml` directly. The equivalent start command is:

```text
gunicorn app:app --workers 1 --threads 1 --timeout 180 --graceful-timeout 30 --max-requests 20 --max-requests-jitter 5
```

Set `SECRET_KEY` and `DATABASE_URL` in the Render environment. `YOLO_MODEL_PATH` and `SWIN_MODEL_PATH` are optional and default to the two model files in the project root. `SWIN_CLASS_NAMES` may be set to four comma-separated labels only when the training metadata confirms them; the repository checkpoint contains no label metadata, so the UI uses `Checkpoint class 0` through `Checkpoint class 3` by default.

## Local verification

```powershell
python -m pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000/health` and confirm the response reports `status: healthy`. Startup loads YOLO only. On a positive detection, primitive YOLO boxes are retained, YOLO memory is released, Swin is lazy-loaded in CPU half precision, each crop is classified one at a time, Swin memory is released, and YOLO is restored for the next request. This exclusive lifecycle is enabled by `SWIN_EXCLUSIVE_MEMORY=true` and prevents the two large models from being resident together on the 512 MiB Render worker.

## Verified Swin checkpoint

`model.pth` is a raw timm state dict with 329 tensors. Its shapes match `swin_base_patch4_window7_224` (`embed_dim=128`, depths `2/2/18/2`, 7x7 windows) and its `head.fc` has four outputs. The checkpoint does not contain class names, configuration, or preprocessing metadata. The implementation therefore uses the verified ImageNet preprocessing convention (bicubic resize/crop to 224 and ImageNet normalization) and does not invent medical class names.
