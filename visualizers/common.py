from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def color_for_identifier(identifier: int, seed: int = 51) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(identifier) + int(seed))
    rgb = rng.integers(32, 256, size=3, dtype=np.int32)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def color_for_identifier_rgba(identifier: int, seed: int = 51) -> list[int]:
    red, green, blue = color_for_identifier(identifier, seed=seed)
    return [red, green, blue, 255]


def format_identity_label(
        track_id: int,
        *,
        global_id: Optional[int] = None,
        probability: Optional[float] = None,
) -> str:
    parts = [f"t{int(track_id)}"]
    if global_id is not None:
        parts.append(f"g{int(global_id)}")
    if probability is not None:
        parts.append(f"p={float(probability):.2f}")
    return " ".join(parts)


def _clip_box(frame_shape, box_xyxy) -> Optional[tuple[int, int, int, int]]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = np.asarray(box_xyxy[:4], dtype=np.float32)
    x1i = int(np.clip(np.floor(x1), 0, max(width - 1, 0)))
    y1i = int(np.clip(np.floor(y1), 0, max(height - 1, 0)))
    x2i = int(np.clip(np.ceil(x2), 0, width))
    y2i = int(np.clip(np.ceil(y2), 0, height))
    if x2i <= x1i or y2i <= y1i:
        return None
    return x1i, y1i, x2i, y2i


def _draw_box_with_label(
        image: np.ndarray,
        box_xyxy,
        label: str,
        color_bgr: tuple[int, int, int],
        *,
        thickness: int = 2,
) -> None:
    clipped = _clip_box(image.shape, box_xyxy)
    if clipped is None:
        return

    x1i, y1i, x2i, y2i = clipped
    cv2.rectangle(image, (x1i, y1i), (x2i, y2i), color_bgr, thickness)

    if not label:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    text_top = max(y1i - text_height - baseline - 4, 0)
    text_bottom = min(text_top + text_height + baseline + 4, image.shape[0] - 1)
    text_right = min(x1i + text_width + 8, image.shape[1] - 1)
    cv2.rectangle(image, (x1i, text_top), (text_right, text_bottom), color_bgr, -1)
    text_origin = (x1i + 4, max(text_bottom - baseline - 2, text_height))
    cv2.putText(image, label, text_origin, font, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)


def render_tracking_overlay(
        frame: np.ndarray,
        tracks,
        detections=None,
        *,
        seed: int = 51,
        show_detector_fallback: bool = True,
) -> np.ndarray:
    annotated = frame.copy()
    track_array = np.asarray(tracks if tracks is not None else [], dtype=np.float32)
    if track_array.ndim == 1 and track_array.size:
        track_array = track_array[None, :]
    elif track_array.ndim == 1:
        track_array = np.empty((0, 8), dtype=np.float32)

    matched_detection_indices = set()
    for track in track_array:
        if track.size < 6:
            continue

        track_id = int(track[4])
        confidence = float(track[5])
        label = format_identity_label(track_id)
        label = f"{label} {confidence:.2f}"
        red, green, blue = color_for_identifier(track_id, seed=seed)
        color_bgr = (blue, green, red)
        _draw_box_with_label(annotated, track[:4], label, color_bgr)

        if track.size >= 8:
            det_index = int(track[7])
            if det_index >= 0:
                matched_detection_indices.add(det_index)

    if not show_detector_fallback:
        return annotated

    det_array = np.asarray(detections if detections is not None else [], dtype=np.float32)
    if det_array.ndim == 1 and det_array.size:
        det_array = det_array[None, :]
    elif det_array.ndim == 1:
        det_array = np.empty((0, 6), dtype=np.float32)

    fallback_color = (0, 215, 255)
    for det_index, det in enumerate(det_array):
        if det.size < 5 or det_index in matched_detection_indices:
            continue
        label = f"det {float(det[4]):.2f}"
        _draw_box_with_label(annotated, det[:4], label, fallback_color, thickness=1)

    return annotated
