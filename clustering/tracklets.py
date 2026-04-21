from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from clustering.common import (
    _build_detection_weights,
    _compute_motion_profile,
    _frame_ranges,
    _normalize_rows,
    _parse_clip_start_time,
    _safe_float,
    _smooth_embeddings,
)


def build_tracklets(all_detections: List[Dict[str, Any]]) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    """Group detections by (clip_id, track_id) into tracklets."""
    tracklets = defaultdict(list)
    for d in all_detections:
        tracklets[(str(d["clip_id"]), int(d["track_id"]))] += [d]
    return tracklets


def compute_tracklet_representative(
        embeddings: np.ndarray,
        use_median: bool = False,
        weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Representative embedding from tracklet detections."""
    if len(embeddings) == 0:
        raise ValueError("Tracklet has no embeddings")

    embeds = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if len(w) != len(embeds):
            raise ValueError("weights length must match embeddings length")
        w = np.clip(w, 1e-6, None)
        w /= np.sum(w)
        emb = (embeds * w[:, None]).sum(axis=0)
    elif use_median:
        emb = np.median(embeds, axis=0)
    else:
        emb = embeds.mean(axis=0)

    n = np.linalg.norm(emb)
    return emb if n == 0 else emb / n


def compute_tracklet_info(
        tracklets: Dict[Tuple[str, int], List[Dict[str, Any]]],
        use_median: bool = False,
        min_tracklet_frames: int = 0,
        use_quality_weights: bool = True,
        quality_alpha: float = 0.75,
        smooth_embeddings: bool = True,
        smoothing_window: int = 5,
) -> List[Dict[str, Any]]:
    """Compute per-tracklet features/metadata for clustering."""
    tracklet_info = []
    dropped = 0

    clip_max_frame: Dict[str, int] = defaultdict(int)
    for (clip_id, _), dets in tracklets.items():
        if dets:
            clip_max_frame[clip_id] = max(
                clip_max_frame[clip_id],
                max(int(det["frame_num"]) for det in dets),
            )

    for (clip_id, track_id), dets in sorted(tracklets.items(), key=lambda x: (x[0][0], x[0][1])):
        dets_sorted = sorted(dets, key=lambda d: int(d["frame_num"]))
        embeds = np.array([det["embeddings"] for det in dets_sorted], dtype=np.float32)
        if len(embeds) < min_tracklet_frames:
            dropped += 1
            continue

        all_embeds = _smooth_embeddings(embeds, smoothing_window) if smooth_embeddings else _normalize_rows(embeds)
        weights = _build_detection_weights(dets_sorted, quality_alpha=quality_alpha) if use_quality_weights else None
        rep = compute_tracklet_representative(
            all_embeds,
            use_median=use_median,
            weights=weights,
        )

        frames = [int(det["frame_num"]) for det in dets_sorted]
        start_frame = int(frames[0])
        end_frame = int(frames[-1])
        center_frame = 0.5 * (start_frame + end_frame)
        denom = max(1, int(clip_max_frame.get(clip_id, end_frame)))
        center_ratio = center_frame / float(denom)

        ts_values = []
        for det in dets_sorted:
            ts = _safe_float(det.get("timestamp_sec"))
            if ts is not None:
                ts_values.append(ts)
        rel_center_sec = float(np.median(ts_values)) if ts_values else None
        clip_start_sec = _parse_clip_start_time(clip_id)
        abs_center_sec = None
        if clip_start_sec is not None and rel_center_sec is not None:
            abs_center_sec = clip_start_sec + rel_center_sec

        quality_values = []
        for det in dets_sorted:
            q = _safe_float(det.get("quality"))
            if q is None:
                q = _safe_float(det.get("confidence"))
            if q is not None:
                quality_values.append(q)
        quality_mean = float(np.mean(quality_values)) if quality_values else None

        tracklet_info.append(
            {
                "clip_id": clip_id,
                "track_id": track_id,
                "embedding": rep,
                "all_embeddings": all_embeds,
                "frame_ranges": _frame_ranges(frames),
                "num_frames": len(frames),
                "temporal_center_ratio": center_ratio,
                "relative_time_center_sec": rel_center_sec,
                "absolute_time_center_sec": abs_center_sec,
                "motion_speed": _compute_motion_profile(dets_sorted),
                "quality_mean": quality_mean,
            }
        )

    if dropped:
        print(f"Dropped {dropped} tracklets with < {min_tracklet_frames} frames")
    return tracklet_info
