"""Low-memory YOLOv8 ONNX Runtime inference helpers."""

import gc
import ast
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

MODEL_PATH = Path(os.getenv("YOLO_ONNX_MODEL_PATH", Path(__file__).resolve().parent / "yolov8_model.onnx"))
INPUT_SIZE = int(os.getenv("YOLO_INPUT_SIZE", "320"))
MODEL_STRIDE = int(os.getenv("YOLO_MODEL_STRIDE", "32"))
CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.25"))
NMS_IOU_THRESHOLD = float(os.getenv("YOLO_NMS_IOU_THRESHOLD", "0.45"))
MAX_DETECTIONS = int(os.getenv("YOLO_MAX_DETECTIONS", "10"))

_session = None
_session_lock = threading.Lock()
_class_names = {}


def session_loaded():
    return _session is not None


def available_memory_mb():
    if os.name == "nt":
        return None
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        current = Path("/sys/fs/cgroup/memory.current").read_text().strip()
        if limit != "max":
            return (int(limit) - int(current)) / 1048576
    except (OSError, ValueError):
        return None
    return None


def get_onnx_session():
    """Load one sequential CPU session per worker, without running warmup inference."""
    global _session, _class_names
    if _session is not None:
        return _session
    with _session_lock:
        if _session is not None:
            return _session
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"ONNX model not found at {MODEL_PATH}")
        started = time.perf_counter()
        LOGGER.info("ONNX SESSION LOAD START path=%s available_memory_mb=%s", MODEL_PATH, available_memory_mb())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = True
        _session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        metadata_names = _session.get_modelmeta().custom_metadata_map.get("names", "")
        try:
            parsed_names = ast.literal_eval(metadata_names)
            if isinstance(parsed_names, dict):
                _class_names = {int(key): str(value) for key, value in parsed_names.items()}
        except (SyntaxError, ValueError, TypeError):
            _class_names = {}
        LOGGER.info(
            "ONNX SESSION LOAD COMPLETE elapsed=%.2fs input=%s available_memory_mb=%s",
            time.perf_counter() - started,
            _session.get_inputs()[0].shape,
            available_memory_mb(),
        )
        return _session


def letterbox_rgb(image):
    """Resize RGB while preserving aspect ratio, then pad with YOLO's 114 gray."""
    width, height = image.size
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    source = np.asarray(image, dtype=np.uint8)
    resized = cv2.resize(source, resized_size, interpolation=cv2.INTER_LINEAR)
    pad_width = (MODEL_STRIDE - resized.shape[1] % MODEL_STRIDE) % MODEL_STRIDE
    pad_height = (MODEL_STRIDE - resized.shape[0] % MODEL_STRIDE) % MODEL_STRIDE
    pad_left = int(round(pad_width / 2 - 0.1))
    pad_top = int(round(pad_height / 2 - 0.1))
    pad_x = pad_left
    pad_y = pad_top
    canvas_width = resized.shape[1] + pad_width
    canvas_height = resized.shape[0] + pad_height
    canvas = np.full((canvas_height, canvas_width, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + resized.shape[0], pad_x:pad_x + resized.shape[1]] = resized
    array = canvas.astype(np.float32)
    del source, resized, canvas
    array = np.ascontiguousarray(array.transpose(2, 0, 1)[None] / 255.0)
    return array, scale, pad_x, pad_y


def _box_iou(box, boxes):
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(box_area + areas - intersection, 1e-7)


def _nms(boxes, scores, class_ids):
    keep = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        indices = indices[np.argsort(scores[indices])[::-1]]
        while indices.size:
            current = indices[0]
            keep.append(current)
            if indices.size == 1:
                break
            overlaps = _box_iou(boxes[current], boxes[indices[1:]])
            indices = indices[1:][overlaps <= NMS_IOU_THRESHOLD]
    return np.asarray(keep, dtype=np.int64)


def postprocess(output, original_size, scale, pad_x, pad_y):
    """Decode YOLOv8 raw [x,y,w,h,class scores] output and apply class-aware NMS."""
    predictions = np.asarray(output)
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.ndim != 2:
        raise ValueError(f"Unexpected ONNX output shape: {predictions.shape}")
    if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] <= 128:
        predictions = predictions.transpose(1, 0)
    if predictions.shape[1] < 5:
        raise ValueError(f"Unexpected YOLO output shape: {predictions.shape}")

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1).astype(np.int64)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    valid = scores >= CONFIDENCE_THRESHOLD
    if not np.any(valid):
        return []

    boxes_xywh = boxes_xywh[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]
    center_x, center_y, width, height = boxes_xywh.T
    boxes = np.column_stack((center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2))
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    original_width, original_height = original_size
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_height)
    keep = _nms(boxes, scores, class_ids)
    keep = keep[np.argsort(scores[keep])[::-1]][:MAX_DETECTIONS]
    return [
        {
            "raw_bbox": [float(value) for value in boxes[index]],
            "yolo_class_id": int(class_ids[index]),
            "yolo_class": _class_names.get(int(class_ids[index]), str(int(class_ids[index]))),
            "yolo_confidence": float(scores[index]),
        }
        for index in keep
    ]


def predict(image):
    session = get_onnx_session()
    tensor, scale, pad_x, pad_y = letterbox_rgb(image)
    started = time.perf_counter()
    try:
        output = session.run(None, {session.get_inputs()[0].name: tensor})
        return postprocess(output[0], image.size, scale, pad_x, pad_y)
    finally:
        del tensor
        if "output" in locals():
            del output
        gc.collect()
        LOGGER.info("ONNX INFERENCE COMPLETE elapsed=%.2fs available_memory_mb=%s", time.perf_counter() - started, available_memory_mb())
