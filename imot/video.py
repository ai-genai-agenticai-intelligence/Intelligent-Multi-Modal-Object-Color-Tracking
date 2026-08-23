from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class VideoSource:
    cap: cv2.VideoCapture
    tmp_path: Optional[str] = None

    def release(self) -> None:
        try:
            self.cap.release()
        finally:
            if self.tmp_path:
                try:
                    os.unlink(self.tmp_path)
                except Exception:
                    pass


def open_uploaded_video(uploaded_bytes: bytes, filename: str) -> VideoSource:
    suffix = os.path.splitext(filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_bytes)
        tmp_path = f.name
    cap = cv2.VideoCapture(tmp_path)
    return VideoSource(cap=cap, tmp_path=tmp_path)


def open_webcam(index: int) -> VideoSource:
    cap = cv2.VideoCapture(int(index))
    return VideoSource(cap=cap, tmp_path=None)


def iter_frames(
    cap: cv2.VideoCapture,
    *,
    stride: int = 1,
    max_frames: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    stride = max(1, int(stride))
    max_frames = int(max_frames)

    frame_i = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        if (frame_i - 1) % stride != 0:
            continue
        yield frame_i - 1, frame
        processed += 1
        if max_frames > 0 and processed >= max_frames:
            break

