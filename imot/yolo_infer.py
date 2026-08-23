from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


YoloTask = Literal["detect", "segment", "pose"]


@dataclass(frozen=True)
class YoloOutput:
    annotated_bgr: np.ndarray
    num_objects: int


def default_weights(task: YoloTask) -> str:
    if task == "segment":
        return "yolov8n-seg.pt"
    if task == "pose":
        return "yolov8n-pose.pt"
    return "yolov8n.pt"


def run_yolo(bgr_frame: np.ndarray, task: YoloTask, weights_path: str) -> YoloOutput:
    # Lazy import keeps Streamlit startup fast if YOLO tab isn't used.
    from ultralytics import YOLO

    model = YOLO(weights_path)
    results = model(bgr_frame)
    annotated = results[0].plot()  # BGR
    num = int(len(results[0].boxes)) if getattr(results[0], "boxes", None) is not None else 0
    return YoloOutput(annotated_bgr=annotated, num_objects=num)

