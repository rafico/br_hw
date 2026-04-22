from __future__ import annotations

import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_detection_cache_path(dataset_dir: str, detections_cache: str = "") -> Path:
    if detections_cache:
        path = Path(detections_cache).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"detections cache not found: {path}")
        return path

    cache_dir = Path(dataset_dir).expanduser().resolve() / ".cache"
    candidates = sorted(cache_dir.glob("*_all_detections*.json"))
    if not candidates:
        raise FileNotFoundError(f"no detections cache found under {cache_dir}")

    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def build_catalogue_lookup(catalogue_payload: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for global_id, appearances in catalogue_payload.get("catalogue", {}).items():
        global_id_int = int(global_id)
        for appearance in appearances:
            frame_ranges = []
            for frame_range in appearance.get("frame_ranges", []):
                if not isinstance(frame_range, list) or len(frame_range) != 2:
                    continue
                frame_ranges.append((int(frame_range[0]), int(frame_range[1])))

            lookup[(str(appearance["clip_id"]), int(appearance["local_track_id"]))] = {
                "global_id": global_id_int,
                "cluster_probability": _safe_float(appearance.get("cluster_probability")),
                "frame_ranges": frame_ranges,
            }
    return lookup


def build_scene_lookup(scene_payload: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("clip_id", "")): item for item in scene_payload}


def group_detections_by_clip_frame(detections: Iterable[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for detection in detections:
        clip_id = str(detection["clip_id"])
        frame_num = int(detection["frame_num"])
        grouped.setdefault(clip_id, {}).setdefault(frame_num, []).append(detection)
    return grouped


def _sample_sorted_values(values: Iterable[int], count: int) -> list[int]:
    unique_values = sorted({int(value) for value in values})
    if count <= 0 or not unique_values:
        return []
    if count >= len(unique_values):
        return unique_values
    if count == 1:
        return [unique_values[len(unique_values) // 2]]

    sampled_indices: set[int] = set()
    max_index = len(unique_values) - 1
    for sample_index in range(count):
        target_index = round(sample_index * max_index / (count - 1))
        for offset in range(len(unique_values)):
            right = target_index + offset
            left = target_index - offset
            if right <= max_index and right not in sampled_indices:
                sampled_indices.add(right)
                break
            if left >= 0 and left not in sampled_indices:
                sampled_indices.add(left)
                break

    return [unique_values[index] for index in sorted(sampled_indices)]


def select_review_frame_keys(
    grouped: dict[str, dict[int, list[dict[str, Any]]]],
    sample_count: int,
) -> list[tuple[str, int]]:
    if sample_count <= 0:
        return []

    clips = [clip_id for clip_id in sorted(grouped) if grouped[clip_id]]
    if not clips:
        return []

    allocations = {clip_id: 0 for clip_id in clips}
    total_allocated = 0

    for clip_id in clips:
        if total_allocated >= sample_count:
            break
        allocations[clip_id] = 1
        total_allocated += 1

    clip_order = sorted(clips, key=lambda clip_id: (-len(grouped[clip_id]), clip_id))
    while total_allocated < sample_count:
        progress = False
        for clip_id in clip_order:
            if allocations[clip_id] >= len(grouped[clip_id]):
                continue
            allocations[clip_id] += 1
            total_allocated += 1
            progress = True
            if total_allocated >= sample_count:
                break
        if not progress:
            break

    sampled_by_clip: dict[str, list[int]] = {}
    for clip_id in clips:
        frame_numbers = sorted(grouped[clip_id])
        sampled_by_clip[clip_id] = _sample_sorted_values(frame_numbers, allocations[clip_id])

    selected: list[tuple[str, int]] = []
    sample_index = 0
    while True:
        progress = False
        for clip_id in clips:
            sampled = sampled_by_clip[clip_id]
            if sample_index >= len(sampled):
                continue
            selected.append((clip_id, sampled[sample_index]))
            progress = True
        if not progress:
            break
        sample_index += 1

    return selected


def _frame_in_ranges(frame_num: int, frame_ranges: Iterable[tuple[int, int]]) -> bool:
    for start, end in frame_ranges:
        if int(start) <= int(frame_num) <= int(end):
            return True
    return False


def _normalize_box(box_xyxy_abs: Any) -> list[float] | None:
    if not isinstance(box_xyxy_abs, list) or len(box_xyxy_abs) != 4:
        return None
    values = []
    for item in box_xyxy_abs:
        value = _safe_float(item)
        if value is None:
            return None
        values.append(value)
    return values


def _overlapping_scene_segments(scene: dict[str, Any] | None, timestamp_sec: float | None) -> list[dict[str, Any]]:
    if scene is None or timestamp_sec is None:
        return []

    overlaps = []
    for segment in scene.get("crime_segments", []) or []:
        start = _safe_float(segment.get("timestamp_start"))
        end = _safe_float(segment.get("timestamp_end"))
        if start is None or end is None or not (start <= timestamp_sec <= end):
            continue
        overlaps.append(
            {
                "timestamp_start": start,
                "timestamp_end": end,
                "involved_people_global": list(segment.get("involved_people_global", [])),
            }
        )
    return overlaps


def build_review_manifest(
    *,
    dataset_dir: str,
    detections: list[dict[str, Any]],
    catalogue_payload: dict[str, Any],
    scene_payload: list[dict[str, Any]],
    detections_cache_path: str,
    catalogue_path: str,
    scene_path: str,
    rerun_recording_path: str,
    sample_count: int = 10,
    consistency_count: int = 5,
) -> dict[str, Any]:
    grouped = group_detections_by_clip_frame(detections)
    catalogue_lookup = build_catalogue_lookup(catalogue_payload)
    scene_lookup = build_scene_lookup(scene_payload)
    frame_keys = select_review_frame_keys(grouped, sample_count)

    warnings: list[str] = []
    review_frames = []

    for clip_id, frame_num in frame_keys:
        detections_for_frame = sorted(grouped[clip_id][frame_num], key=lambda item: int(item["track_id"]))
        timestamp_sec = None
        for detection in detections_for_frame:
            timestamp_sec = _safe_float(detection.get("timestamp_sec"))
            if timestamp_sec is not None:
                break

        scene = scene_lookup.get(clip_id)
        if scene is None:
            warnings.append(f"scene payload does not contain clip {clip_id!r}")

        detections_out = []
        track_ids = []
        global_ids = []
        for detection in detections_for_frame:
            track_id = int(detection["track_id"])
            track_ids.append(track_id)
            catalogue_entry = catalogue_lookup.get((clip_id, track_id))

            global_id = None
            cluster_probability = None
            catalogue_frame_match = False
            if catalogue_entry is None:
                warnings.append(f"catalogue is missing mapping for {clip_id}:t{track_id}")
            else:
                global_id = int(catalogue_entry["global_id"])
                global_ids.append(global_id)
                cluster_probability = catalogue_entry.get("cluster_probability")
                catalogue_frame_match = _frame_in_ranges(frame_num, catalogue_entry["frame_ranges"])
                if not catalogue_frame_match:
                    warnings.append(
                        f"frame {clip_id}:{frame_num} falls outside catalogue ranges for track {track_id}"
                    )

            detections_out.append(
                {
                    "track_id": track_id,
                    "global_id": global_id,
                    "cluster_probability": cluster_probability,
                    "confidence": _safe_float(detection.get("confidence")),
                    "box_xyxy_abs": _normalize_box(detection.get("box_xyxy_abs")),
                    "catalogue_frame_match": catalogue_frame_match,
                }
            )

        review_frames.append(
            {
                "clip_id": clip_id,
                "frame_num": frame_num,
                "timestamp_sec": timestamp_sec,
                "scene_label": None if scene is None else scene.get("label"),
                "scene_segments_overlapping_frame": _overlapping_scene_segments(scene, timestamp_sec),
                "track_ids": track_ids,
                "global_ids": sorted(set(global_ids)),
                "detections": detections_out,
            }
        )

    dataset_dir_abs = str(Path(dataset_dir).expanduser().resolve())
    rerun_recording_abs = str(Path(rerun_recording_path).expanduser().resolve())
    videos_arg = shlex.quote(dataset_dir_abs)
    rerun_arg = shlex.quote(rerun_recording_abs)

    consistency_frames = [
        {
            "clip_id": entry["clip_id"],
            "frame_num": entry["frame_num"],
            "timestamp_sec": entry["timestamp_sec"],
        }
        for entry in review_frames[: max(0, min(consistency_count, len(review_frames)))]
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "dataset_dir": dataset_dir_abs,
            "detections_cache": str(Path(detections_cache_path).expanduser().resolve()),
            "catalogue": str(Path(catalogue_path).expanduser().resolve()),
            "scene": str(Path(scene_path).expanduser().resolve()),
            "rerun_recording": rerun_recording_abs,
        },
        "review_commands": {
            "opencv": f"python run.py --dataset-dir {videos_arg} --show --visualizer none",
            "fiftyone": f"python run.py --dataset-dir {videos_arg} --visualizer fiftyone",
            "rerun": f"python run.py --dataset-dir {videos_arg} --visualizer rerun --rerun-save {rerun_arg}",
        },
        "summary": {
            "review_frame_count": len(review_frames),
            "consistency_frame_count": len(consistency_frames),
            "clip_count": len({entry['clip_id'] for entry in review_frames}),
        },
        "review_frames": review_frames,
        "consistency_frames": consistency_frames,
        "warnings": list(dict.fromkeys(warnings)),
    }


def render_review_notes(manifest: dict[str, Any]) -> str:
    artifacts = manifest["artifacts"]
    commands = manifest["review_commands"]
    review_frames = manifest["review_frames"]
    consistency_frames = manifest["consistency_frames"]

    lines = [
        "# Manual Visual Review Notes",
        "",
        f"- Generated: `{manifest['generated_at']}`",
        f"- Dataset: `{artifacts['dataset_dir']}`",
        f"- Detections cache: `{artifacts['detections_cache']}`",
        f"- Catalogue: `{artifacts['catalogue']}`",
        f"- Scene labels: `{artifacts['scene']}`",
        f"- Rerun recording: `{artifacts['rerun_recording']}`",
        "",
        "## Review Commands",
        "",
        f"- OpenCV: `{commands['opencv']}`",
        f"- FiftyOne: `{commands['fiftyone']}`",
        f"- Rerun: `{commands['rerun']}`",
        "",
        "## Sampled Frames",
        "",
        "| Clip | Frame | Timestamp (s) | Tracks | Global IDs |",
        "| --- | ---: | ---: | --- | --- |",
    ]

    for entry in review_frames:
        timestamp = "n/a" if entry["timestamp_sec"] is None else f"{entry['timestamp_sec']:.2f}"
        tracks = ", ".join(str(track_id) for track_id in entry["track_ids"]) or "none"
        global_ids = ", ".join(str(global_id) for global_id in entry["global_ids"]) or "none"
        lines.append(f"| {entry['clip_id']} | {entry['frame_num']} | {timestamp} | {tracks} | {global_ids} |")

    lines.extend(
        [
            "",
            "## OpenCV Live View",
            "",
            "- [ ] Boxes appear on the first visible detections, not only after track warm-up.",
            "- [ ] Track IDs stay readable and move with the correct person.",
            "- [ ] Detector fallback boxes appear before track maturity when needed.",
            "- [ ] The window exits cleanly with `q`.",
            "",
            "## FiftyOne",
            "",
            "- [ ] Sample at least 10 frames across multiple clips.",
            "- [ ] Stored detections align with visible person crops.",
            "- [ ] Sampled frames match the cached detections for IDs and boxes.",
            "",
            "## Rerun",
            "",
            "- [ ] Inspect the same sampled frames in the `.rrd` recording.",
            "- [ ] Frame time, detection boxes, labels, and colors are aligned.",
            "- [ ] Post-clustering labels use global IDs where available.",
            "- [ ] Scene summaries reference the expected clip and timestamps.",
            "",
            "## Cross-Artifact Consistency Frames",
            "",
        ]
    )

    for entry in consistency_frames:
        timestamp = "n/a" if entry["timestamp_sec"] is None else f"{entry['timestamp_sec']:.2f}s"
        lines.append(f"- [ ] {entry['clip_id']} frame {entry['frame_num']} ({timestamp})")

    if manifest["warnings"]:
        lines.extend(
            [
                "",
                "## Prep Warnings",
                "",
            ]
        )
        for warning in manifest["warnings"]:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Reviewer Notes",
            "",
            "- Reviewer:",
            "- Commit:",
            "- Result:",
            "- Notes:",
        ]
    )

    return "\n".join(lines) + "\n"


def load_review_inputs(
    *,
    dataset_dir: str,
    detections_cache: str,
    catalogue: str,
    scene: str,
    rerun_recording: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], Path]:
    detections_cache_path = resolve_detection_cache_path(dataset_dir, detections_cache=detections_cache)
    catalogue_path = Path(catalogue).expanduser().resolve()
    scene_path = Path(scene).expanduser().resolve()
    rerun_path = Path(rerun_recording).expanduser().resolve()

    for path, label in (
        (catalogue_path, "catalogue"),
        (scene_path, "scene labels"),
        (rerun_path, "Rerun recording"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    detections = _load_json(detections_cache_path)
    catalogue_payload = _load_json(catalogue_path)
    scene_payload = _load_json(scene_path)
    return detections, catalogue_payload, scene_payload, detections_cache_path
