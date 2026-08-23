from __future__ import annotations

import cv2
import numpy as np


def preset_hsv(name: str) -> tuple[np.ndarray, np.ndarray]:
    name = name.lower().strip()
    if name == "red":
        return np.array([161, 155, 84], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)
    if name == "blue":
        return np.array([94, 80, 2], dtype=np.uint8), np.array([126, 255, 255], dtype=np.uint8)
    if name == "green":
        return np.array([40, 100, 100], dtype=np.uint8), np.array([102, 255, 255], dtype=np.uint8)
    if name in {"every color except white", "non-white"}:
        return np.array([0, 42, 0], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)
    return np.array([0, 0, 0], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)


def mask_and_stats(bgr_frame: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low, high)

    mask_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    coverage = float(mask_pixels / total_pixels) if total_pixels else 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [float(cv2.contourArea(c)) for c in contours]

    return {
        "mask": mask,
        "coverage": coverage,
        "num_contours": int(len(contours)),
        "largest_contour_area": float(max(areas)) if areas else 0.0,
    }


def overlay_red_mask(bgr_frame: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    overlay = bgr_frame.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    overlay[mask > 0] = cv2.addWeighted(overlay[mask > 0], 1 - alpha, red[mask > 0], alpha, 0)
    return overlay

