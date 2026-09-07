"""Export the trained YOLO detector for the production ONNX Runtime path."""

import argparse
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Export yolov8_model.pt to ONNX.")
    parser.add_argument("--model", type=Path, default=BASE_DIR / "yolov8_model.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "yolov8_model.onnx")
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO source model not found: {model_path}")

    model = YOLO(str(model_path))
    exported_path = Path(model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=12,
        simplify=False,
        dynamic=True,
        half=False,
        device="cpu",
        nms=False,
    ))
    if exported_path.resolve() != args.output.resolve():
        args.output.write_bytes(exported_path.read_bytes())
        exported_path.unlink(missing_ok=True)
        exported_path = args.output
    print(f"Exported ONNX model: {exported_path.resolve()}")
    print(f"ONNX model size: {exported_path.stat().st_size / 1048576:.2f} MB")


if __name__ == "__main__":
    main()
