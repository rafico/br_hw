from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _frame_ranges(frame_list: List[int]) -> List[List[int]]:
    """Convert sorted list of frame numbers into [start, end] ranges."""
    if not frame_list:
        return []
    ranges, start, prev = [], frame_list[0], frame_list[0]
    for f in frame_list[1:]:
        if f == prev + 1:
            prev = f
        else:
            ranges.append([start, prev])
            start = prev = f
    ranges.append([start, prev])
    return ranges


def _check_distance_matrix(D: np.ndarray):
    """Basic sanity checks to catch issues early."""
    if not np.isfinite(D).all():
        raise ValueError("Distance matrix contains non-finite values.")
    if (D < 0).any():
        raise ValueError("Distance matrix has negative entries.")
    if not np.allclose(D, D.T, atol=1e-6):
        raise ValueError("Distance matrix must be symmetric.")


def tracklets_cooccur(t1: dict, t2: dict) -> bool:
    if str(t1["clip_id"]) != str(t2["clip_id"]):
        return False
    for a1, a2 in t1["frame_ranges"]:
        for b1, b2 in t2["frame_ranges"]:
            if a1 <= b2 and b1 <= a2:
                return True
    return False


def _components_cooccur(
        members_a: List[int],
        members_b: List[int],
        tracklet_info: List[Dict[str, Any]],
) -> bool:
    return any(
        tracklets_cooccur(tracklet_info[i], tracklet_info[j])
        for i in members_a
        for j in members_b
    )


def _parse_clip_start_time(clip_id: str) -> Optional[float]:
    """Best-effort parse of absolute start timestamp from clip name."""
    s = str(clip_id)

    dt_patterns = [
        r"(20\d{2})(\d{2})(\d{2})[_\-T]?(\d{2})(\d{2})(\d{2})",
        r"(20\d{2})[-_](\d{2})[-_](\d{2})[_\-T](\d{2})[-_](\d{2})[-_](\d{2})",
    ]
    for pattern in dt_patterns:
        m = re.search(pattern, s)
        if not m:
            continue
        try:
            parts = [int(g) for g in m.groups()]
            dt = datetime(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
            return dt.timestamp()
        except ValueError:
            continue

    ms_match = re.search(r"(?<!\d)(1\d{12})(?!\d)", s)
    if ms_match:
        return int(ms_match.group(1)) / 1000.0
    sec_match = re.search(r"(?<!\d)(1\d{9})(?!\d)", s)
    if sec_match:
        return float(int(sec_match.group(1)))

    return None


def _smooth_embeddings(embeddings: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Simple moving-average smoothing along time for tracklet embeddings."""
    embeds = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
    n = len(embeds)
    if n < 3 or window_size <= 1:
        return embeds

    w = max(1, int(window_size))
    if w % 2 == 0:
        w += 1
    half = w // 2

    smoothed = np.empty_like(embeds)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed[i] = embeds[lo:hi].mean(axis=0)
    return _normalize_rows(smoothed)


def _build_detection_weights(
        dets: List[Dict[str, Any]],
        quality_alpha: float = 0.75,
) -> np.ndarray:
    """Build representative-aggregation weights using quality/confidence."""
    alpha = float(np.clip(quality_alpha, 0.0, 1.0))
    raw = []
    for det in dets:
        q = _safe_float(det.get("quality"))
        if q is None:
            q = _safe_float(det.get("confidence"))
        if q is None:
            q = 1.0
        raw.append(float(np.clip(q, 0.0, 1.0)))

    quality = np.asarray(raw, dtype=np.float32)
    weights = (1.0 - alpha) + (alpha * quality)
    weights = np.clip(weights, 1e-6, None)
    weights /= np.sum(weights)
    return weights


def _compute_motion_profile(dets: List[Dict[str, Any]]) -> Optional[float]:
    """Estimate motion speed from normalized bbox centers (if available)."""
    frames = []
    centers = []
    for det in dets:
        cx = _safe_float(det.get("center_x"))
        cy = _safe_float(det.get("center_y"))
        if cx is None or cy is None:
            continue
        frames.append(int(det["frame_num"]))
        centers.append((cx, cy))

    if len(frames) < 2:
        return None

    speeds = []
    for idx in range(1, len(frames)):
        dt = max(1, int(frames[idx] - frames[idx - 1]))
        dx = centers[idx][0] - centers[idx - 1][0]
        dy = centers[idx][1] - centers[idx - 1][1]
        speeds.append(float(np.hypot(dx, dy) / dt))

    if not speeds:
        return None
    return float(np.median(np.asarray(speeds, dtype=np.float32)))
