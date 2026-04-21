from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, v_measure_score

LOGGER = logging.getLogger(__name__)


def _expand_ranges(frame_ranges: Iterable[Iterable[int]]) -> set[Tuple[str, int]]:
    expanded = set()
    for start, end in frame_ranges:
        for frame_num in range(int(start), int(end) + 1):
            expanded.add(frame_num)
    return expanded


def _person_frame_sets_from_catalogue(catalogue: dict, clip_key: str = "clip_id") -> Dict[int, set]:
    person_frames: Dict[int, set] = {}
    for person_id, appearances in catalogue.items():
        frames = set()
        for appearance in appearances:
            clip_id = str(appearance[clip_key])
            for start, end in appearance.get("frame_ranges", []):
                for frame_num in range(int(start), int(end) + 1):
                    frames.add((clip_id, frame_num))
        person_frames[int(person_id)] = frames
    return person_frames


def _person_frame_sets_from_gt(persons: list) -> Dict[int, set]:
    person_frames: Dict[int, set] = {}
    for person in persons:
        frames = set()
        for appearance in person.get("appearances", []):
            clip_id = str(appearance["clip"])
            for start, end in appearance.get("frame_ranges", []):
                for frame_num in range(int(start), int(end) + 1):
                    frames.add((clip_id, frame_num))
        person_frames[int(person["global_id"])] = frames
    return person_frames


def _hungarian_match(gt_frames: Dict[int, set], pred_frames: Dict[int, set]) -> Dict[int, Optional[int]]:
    if not gt_frames or not pred_frames:
        return {pred_id: None for pred_id in pred_frames}

    gt_ids = sorted(gt_frames)
    pred_ids = sorted(pred_frames)
    overlap = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for i, gt_id in enumerate(gt_ids):
        for j, pred_id in enumerate(pred_ids):
            overlap[i, j] = float(len(gt_frames[gt_id] & pred_frames[pred_id]))

    row_ind, col_ind = linear_sum_assignment(-overlap)
    matched = {pred_id: None for pred_id in pred_ids}
    for row, col in zip(row_ind, col_ind):
        if overlap[row, col] > 0:
            matched[pred_ids[col]] = gt_ids[row]
    return matched


def _labels_for_items(gt_frames: Dict[int, set], pred_frames: Dict[int, set]) -> Tuple[list, list]:
    universe = sorted(set().union(*gt_frames.values(), *pred_frames.values()) if (gt_frames or pred_frames) else [])
    gt_lookup = {}
    pred_lookup = {}
    for gt_id, frames in gt_frames.items():
        for item in frames:
            gt_lookup[item] = gt_id
    for pred_id, frames in pred_frames.items():
        for item in frames:
            pred_lookup[item] = pred_id

    gt_labels = [gt_lookup.get(item, -1) for item in universe]
    pred_labels = [pred_lookup.get(item, -1) for item in universe]
    return gt_labels, pred_labels


def _cluster_purity(gt_labels: list, pred_labels: list) -> float:
    if not gt_labels:
        return 0.0
    total = 0
    pred_clusters = sorted(set(pred_labels))
    for cluster_id in pred_clusters:
        cluster_gt = [gt for gt, pred in zip(gt_labels, pred_labels) if pred == cluster_id]
        if not cluster_gt:
            continue
        counts = np.bincount(np.asarray(cluster_gt, dtype=np.int64) + 1)
        total += int(np.max(counts))
    return float(total / max(len(gt_labels), 1))


def _interval_iou(gt_spans: List[List[float]], pred_spans: List[List[float]], resolution: float = 0.1) -> float:
    if not gt_spans and not pred_spans:
        return 1.0
    max_t = 0.0
    for spans in (gt_spans, pred_spans):
        for start, end in spans:
            max_t = max(max_t, float(end))
    if max_t <= 0.0:
        return 0.0

    ticks = np.arange(0.0, max_t + resolution, resolution)
    gt_mask = np.zeros_like(ticks, dtype=bool)
    pred_mask = np.zeros_like(ticks, dtype=bool)
    for start, end in gt_spans:
        gt_mask |= (ticks >= float(start)) & (ticks <= float(end))
    for start, end in pred_spans:
        pred_mask |= (ticks >= float(start)) & (ticks <= float(end))
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(gt_mask, pred_mask).sum() / union)


def _scene_metrics(gt_scenes: list, pred_scenes: list) -> dict:
    pred_by_clip = {str(item["clip_id"]): item for item in pred_scenes}
    gt_labels = []
    pred_labels = []
    per_clip = {}

    for scene in gt_scenes:
        clip_id = str(scene["clip"])
        pred = pred_by_clip.get(clip_id, {})
        gt_labels.append(scene["label"])
        pred_labels.append(pred.get("label", "missing"))
        gt_spans = scene.get("crime_spans_sec", [])
        pred_spans = [
            [segment.get("timestamp_start", 0.0), segment.get("timestamp_end", 0.0)]
            for segment in pred.get("crime_segments", [])
        ]
        pred_people = {
            int(pid)
            for segment in pred.get("crime_segments", [])
            for pid in segment.get("involved_people_global", [])
        }
        gt_people = {int(pid) for pid in scene.get("crime_person_global_ids", [])}
        per_clip[clip_id] = {
            "label": pred.get("label"),
            "crime_span_iou": _interval_iou(gt_spans, pred_spans),
            "crime_person_recall": (
                float(len(gt_people & pred_people) / max(len(gt_people), 1))
                if gt_people
                else 1.0
            ),
        }

    return {
        "accuracy": float(accuracy_score(gt_labels, pred_labels)) if gt_labels else 0.0,
        "macro_f1": float(f1_score(gt_labels, pred_labels, average="macro")) if gt_labels else 0.0,
        "per_clip": per_clip,
    }


def run(
    gt_path: str = "ground_truth.json",
    pred_catalogue: str = "catalogue_v2.json",
    pred_scene: str = "scene_labels_v2.json",
    output: str = "eval_report.json",
) -> Optional[dict]:
    if not Path(gt_path).exists():
        LOGGER.warning("Ground truth file is missing: %s", gt_path)
        return None

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(pred_catalogue, "r", encoding="utf-8") as f:
        pred_catalogue_payload = json.load(f)
    with open(pred_scene, "r", encoding="utf-8") as f:
        pred_scenes = json.load(f)

    gt_frames = _person_frame_sets_from_gt(gt.get("persons", []))
    pred_frames = _person_frame_sets_from_catalogue(pred_catalogue_payload.get("catalogue", {}))
    matched = _hungarian_match(gt_frames, pred_frames)
    gt_labels, pred_labels = _labels_for_items(gt_frames, pred_frames)

    per_id = {}
    for pred_id, gt_id in matched.items():
        if gt_id is None:
            continue
        pred_set = pred_frames[pred_id]
        gt_set = gt_frames[gt_id]
        overlap = len(pred_set & gt_set)
        precision = overlap / max(len(pred_set), 1)
        recall = overlap / max(len(gt_set), 1)
        f1 = (2 * precision * recall / max(precision + recall, 1e-12))
        per_id[str(gt_id)] = {
            "matched_pred_id": pred_id,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    report = {
        "person_reid": {
            "v_measure": float(v_measure_score(gt_labels, pred_labels)) if gt_labels else 0.0,
            "adjusted_rand_index": float(adjusted_rand_score(gt_labels, pred_labels)) if gt_labels else 0.0,
            "purity": _cluster_purity(gt_labels, pred_labels),
            "matched_ids": matched,
            "per_gt_id": per_id,
        },
        "scene": _scene_metrics(gt.get("scenes", []), pred_scenes),
    }

    output_path = Path(output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = output_path.with_suffix(".md")
    md_path.write_text(
        (
            "| Metric | Value |\n"
            "|---|---:|\n"
            "| V-measure | {v_measure:.4f} |\n"
            "| ARI | {ari:.4f} |\n"
            "| Purity | {purity:.4f} |\n"
            "| Scene Accuracy | {scene_acc:.4f} |\n"
            "| Scene Macro-F1 | {scene_f1:.4f} |\n"
        ).format(
            v_measure=report["person_reid"]["v_measure"],
            ari=report["person_reid"]["adjusted_rand_index"],
            purity=report["person_reid"]["purity"],
            scene_acc=report["scene"]["accuracy"],
            scene_f1=report["scene"]["macro_f1"],
        ),
        encoding="utf-8",
    )
    LOGGER.info("Evaluation report saved to %s", output_path.resolve())
    return report


if __name__ == "__main__":
    run()
