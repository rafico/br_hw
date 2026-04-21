from __future__ import annotations

import importlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from .projection import project_2d


def _load_rerun():
    try:
        rr = importlib.import_module("rerun")
    except ImportError as exc:
        raise RuntimeError(
            "Rerun export requires the optional `rerun-sdk` dependency. "
            "Install it with `pip install -r requirements-rerun.txt`."
        ) from exc
    return rr


def _set_time_sequence(rr, timeline: str, sequence: int) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence(timeline, int(sequence))
    else:
        rr.set_time(timeline, sequence=int(sequence))


def _set_time_seconds(rr, timeline: str, seconds: Optional[float]) -> None:
    if seconds is None:
        return
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds(timeline, float(seconds))
    else:
        rr.set_time(timeline, seconds=float(seconds))


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def _color_for_id(identifier: int, seed: int = 51) -> list[int]:
    rng = np.random.default_rng(int(identifier) + int(seed))
    rgb = rng.integers(32, 256, size=3, dtype=np.int32)
    return [int(rgb[0]), int(rgb[1]), int(rgb[2]), 255]


def _build_global_lookup(catalogue_payload: dict) -> Tuple[Dict[Tuple[str, int], int], Dict[Tuple[str, int], float]]:
    global_ids: Dict[Tuple[str, int], int] = {}
    probabilities: Dict[Tuple[str, int], float] = {}
    for gid, appearances in catalogue_payload.get("catalogue", {}).items():
        gid_int = int(gid)
        for appearance in appearances:
            key = (str(appearance["clip_id"]), int(appearance["local_track_id"]))
            global_ids[key] = gid_int
            probabilities[key] = float(appearance.get("cluster_probability", 0.0) or 0.0)
    return global_ids, probabilities


def _group_detections_by_clip_frame(detections: Iterable[dict]) -> Dict[str, Dict[int, list[dict]]]:
    grouped: Dict[str, Dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for det in detections:
        grouped[str(det["clip_id"])][int(det["frame_num"])].append(det)
    return grouped


def _load_rgb_frame(video_path: str, frame_num: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(frame_num) - 1, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _scene_text(scene: dict) -> str:
    lines = [
        f"clip: {scene.get('clip_id', '')}",
        f"label: {scene.get('label', 'unknown')}",
        f"confidence: {scene.get('confidence', scene.get('max_confidence', 'n/a'))}",
    ]
    rationale = scene.get("rationale") or scene.get("justification")
    if rationale:
        lines.append("")
        lines.append(str(rationale))

    segments = scene.get("crime_segments", [])
    if segments:
        lines.append("")
        lines.append("crime_segments:")
        for segment in segments:
            people = ", ".join(str(pid) for pid in segment.get("involved_people_global", [])) or "none"
            lines.append(
                "- "
                f"{segment.get('timestamp_start', 0.0):.2f}s-{segment.get('timestamp_end', 0.0):.2f}s "
                f"people={people}"
            )
    return "\n".join(lines)


def _log_metrics(rr, catalogue_payload: dict) -> None:
    summary = catalogue_payload.get("summary", {})
    adaptive = summary.get("adaptive", {})
    _set_time_sequence(rr, "step", 0)

    if "total_unique_persons" in summary:
        rr.log("metrics/total_unique_persons", rr.Scalar(float(summary["total_unique_persons"])))
    if "total_tracklets" in summary:
        rr.log("metrics/total_tracklets", rr.Scalar(float(summary["total_tracklets"])))
    if "epsilon" in adaptive:
        rr.log("metrics/epsilon", rr.Scalar(float(adaptive["epsilon"])))
    if "gate" in adaptive:
        rr.log("metrics/gate", rr.Scalar(float(adaptive["gate"])))

    rr.log("metrics/summary", rr.TextDocument(str(summary)))


def _log_frame_samples(
    rr,
    grouped: Dict[str, Dict[int, list[dict]]],
    sample_every: int,
    max_frames_per_clip: Optional[int],
) -> None:
    for clip_id, frame_map in grouped.items():
        logged = 0
        for frame_num in sorted(frame_map):
            if sample_every > 1 and ((frame_num - 1) % sample_every) != 0:
                continue
            if max_frames_per_clip is not None and logged >= max_frames_per_clip:
                break

            frame_rgb = _load_rgb_frame(frame_map[frame_num][0]["video_path"], frame_num)
            if frame_rgb is None:
                continue

            dets = frame_map[frame_num]
            timestamp_sec = dets[0].get("timestamp_sec")
            _set_time_sequence(rr, "frame", frame_num)
            _set_time_seconds(rr, "video_time", timestamp_sec)
            rr.log(f"clips/{clip_id}/frame", rr.Image(frame_rgb))
            logged += 1


def _log_detection_boxes(
    rr,
    grouped: Dict[str, Dict[int, list[dict]]],
    global_lookup: Dict[Tuple[str, int], int],
    probability_lookup: Dict[Tuple[str, int], float],
    seed: int,
) -> None:
    for clip_id, frame_map in grouped.items():
        for frame_num, dets in sorted(frame_map.items()):
            boxes = []
            colors = []
            labels = []

            for det in dets:
                key = (str(det["clip_id"]), int(det["track_id"]))
                global_id = global_lookup.get(key)
                color_id = global_id if global_id is not None else int(det["track_id"])
                probability = probability_lookup.get(key)
                boxes.append(np.asarray(det["box_xyxy_abs"], dtype=np.float32))
                colors.append(_color_for_id(color_id, seed=seed))
                label_parts = [f"t{int(det['track_id'])}"]
                if global_id is not None:
                    label_parts.append(f"g{global_id}")
                if probability is not None:
                    label_parts.append(f"p={probability:.2f}")
                labels.append(" ".join(label_parts))

            if not boxes:
                continue

            _set_time_sequence(rr, "frame", frame_num)
            _set_time_seconds(rr, "video_time", dets[0].get("timestamp_sec"))
            rr.log(
                f"clips/{clip_id}/detections",
                rr.Boxes2D(
                    array=np.asarray(boxes, dtype=np.float32),
                    array_format=rr.Box2DFormat.XYXY,
                    colors=colors,
                    labels=labels,
                ),
            )


def _log_embedding_views(
    rr,
    detections: list[dict],
    global_lookup: Dict[Tuple[str, int], int],
    seed: int,
) -> None:
    if not detections:
        return

    det_embeddings = _normalize_rows(np.asarray([det["embeddings"] for det in detections], dtype=np.float32))
    det_coords = project_2d(det_embeddings, seed=seed)
    det_colors = []
    det_labels = []
    for det in detections:
        key = (str(det["clip_id"]), int(det["track_id"]))
        global_id = global_lookup.get(key)
        color_id = global_id if global_id is not None else int(det["track_id"])
        det_colors.append(_color_for_id(color_id, seed=seed))
        det_labels.append(f"{det['clip_id']}:t{int(det['track_id'])}:f{int(det['frame_num'])}")

    rr.log(
        "embeddings/detections_2d",
        rr.Points2D(det_coords, colors=det_colors, labels=det_labels),
    )

    tracklet_vectors = defaultdict(list)
    for det in detections:
        tracklet_vectors[(str(det["clip_id"]), int(det["track_id"]))].append(
            np.asarray(det["embeddings"], dtype=np.float32)
        )

    if not tracklet_vectors:
        return

    tracklet_keys = sorted(tracklet_vectors)
    tracklet_embeddings = _normalize_rows(
        np.asarray(
            [np.mean(np.stack(tracklet_vectors[key], axis=0), axis=0) for key in tracklet_keys],
            dtype=np.float32,
        )
    )
    tracklet_coords = project_2d(tracklet_embeddings, seed=seed)
    tracklet_colors = []
    tracklet_labels = []
    for clip_id, track_id in tracklet_keys:
        global_id = global_lookup.get((clip_id, track_id))
        color_id = global_id if global_id is not None else int(track_id)
        tracklet_colors.append(_color_for_id(color_id, seed=seed))
        label = f"{clip_id}:t{track_id}"
        if global_id is not None:
            label = f"{label}:g{global_id}"
        tracklet_labels.append(label)

    rr.log(
        "embeddings/tracklets_2d",
        rr.Points2D(tracklet_coords, colors=tracklet_colors, labels=tracklet_labels),
    )


def _log_scene_payload(rr, scene_payload: Iterable[dict]) -> None:
    for scene in scene_payload:
        clip_id = str(scene.get("clip_id", "unknown"))
        rr.log(f"scenes/{clip_id}/summary", rr.TextDocument(_scene_text(scene)))


def export_to_rerun(
    *,
    detections,
    catalogue_payload,
    scene_payload,
    output_rrd: Optional[str] = None,
    spawn_viewer: bool = False,
    seed: int = 51,
    sample_every: int = 1,
    max_frames_per_clip: Optional[int] = None,
) -> None:
    rr = _load_rerun()
    rr.init("br_hw_rerun_export")

    if output_rrd:
        output_path = Path(output_rrd)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(output_path))
    if spawn_viewer:
        rr.spawn()

    detections = list(detections or [])
    catalogue_payload = catalogue_payload or {}
    scene_payload = list(scene_payload or [])
    global_lookup, probability_lookup = _build_global_lookup(catalogue_payload)
    grouped = _group_detections_by_clip_frame(detections)

    _log_metrics(rr, catalogue_payload)
    _log_frame_samples(
        rr,
        grouped=grouped,
        sample_every=max(1, int(sample_every)),
        max_frames_per_clip=max_frames_per_clip,
    )
    _log_detection_boxes(rr, grouped, global_lookup, probability_lookup, seed=seed)
    _log_embedding_views(rr, detections, global_lookup, seed=seed)
    _log_scene_payload(rr, scene_payload)
