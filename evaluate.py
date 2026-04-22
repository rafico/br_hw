from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except ImportError:
    _scipy_linear_sum_assignment = None

try:
    from sklearn.metrics import accuracy_score as _sklearn_accuracy_score
    from sklearn.metrics import adjusted_rand_score as _sklearn_adjusted_rand_score
    from sklearn.metrics import f1_score as _sklearn_f1_score
    from sklearn.metrics import v_measure_score as _sklearn_v_measure_score
except ImportError:
    _sklearn_accuracy_score = None
    _sklearn_adjusted_rand_score = None
    _sklearn_f1_score = None
    _sklearn_v_measure_score = None


def _comb2(n: np.ndarray | int | float) -> np.ndarray | float:
    arr = np.asarray(n, dtype=np.float64)
    result = arr * (arr - 1.0) * 0.5
    if result.ndim == 0:
        return float(result)
    return result


def _contingency_matrix(gt_labels: list, pred_labels: list) -> np.ndarray:
    if not gt_labels or not pred_labels:
        return np.zeros((0, 0), dtype=np.float64)

    gt_values = sorted(set(gt_labels))
    pred_values = sorted(set(pred_labels))
    gt_index = {label: idx for idx, label in enumerate(gt_values)}
    pred_index = {label: idx for idx, label in enumerate(pred_values)}
    matrix = np.zeros((len(gt_values), len(pred_values)), dtype=np.float64)
    for gt_label, pred_label in zip(gt_labels, pred_labels):
        matrix[gt_index[gt_label], pred_index[pred_label]] += 1.0
    return matrix


def _accuracy_score(gt_labels: list, pred_labels: list) -> float:
    if _sklearn_accuracy_score is not None:
        return float(_sklearn_accuracy_score(gt_labels, pred_labels))
    if not gt_labels:
        return 0.0
    return float(sum(int(gt == pred) for gt, pred in zip(gt_labels, pred_labels)) / len(gt_labels))


def _macro_f1_score(gt_labels: list, pred_labels: list) -> float:
    if _sklearn_f1_score is not None:
        return float(_sklearn_f1_score(gt_labels, pred_labels, average="macro"))
    labels = sorted(set(gt_labels) | set(pred_labels))
    if not labels:
        return 0.0

    f1_values = []
    for label in labels:
        tp = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == label and pred == label)
        fp = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt != label and pred == label)
        fn = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == label and pred != label)
        if tp == 0 and fp == 0 and fn == 0:
            f1_values.append(0.0)
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append((2.0 * precision * recall) / (precision + recall))
    return float(np.mean(f1_values))


def _adjusted_rand_score(gt_labels: list, pred_labels: list) -> float:
    if _sklearn_adjusted_rand_score is not None:
        return float(_sklearn_adjusted_rand_score(gt_labels, pred_labels))
    n = len(gt_labels)
    if n < 2:
        return 1.0
    contingency = _contingency_matrix(gt_labels, pred_labels)
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    sum_comb_c = float(np.sum(_comb2(contingency)))
    sum_comb_rows = float(np.sum(_comb2(row_sums)))
    sum_comb_cols = float(np.sum(_comb2(col_sums)))
    total_pairs = float(_comb2(n))
    expected_index = (sum_comb_rows * sum_comb_cols) / max(total_pairs, 1e-12)
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
    denom = max_index - expected_index
    if abs(denom) <= 1e-12:
        return 1.0
    return float((sum_comb_c - expected_index) / denom)


def _entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = counts[counts > 0.0] / total
    return float(-np.sum(probs * np.log(probs)))


def _v_measure_score(gt_labels: list, pred_labels: list) -> float:
    if _sklearn_v_measure_score is not None:
        return float(_sklearn_v_measure_score(gt_labels, pred_labels))
    if not gt_labels:
        return 0.0

    contingency = _contingency_matrix(gt_labels, pred_labels)
    if contingency.size == 0:
        return 0.0
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    total = float(contingency.sum())
    h_gt = _entropy(row_sums)
    h_pred = _entropy(col_sums)

    h_gt_given_pred = 0.0
    for col_idx, col_total in enumerate(col_sums):
        if col_total <= 0.0:
            continue
        probs = contingency[:, col_idx]
        probs = probs[probs > 0.0] / col_total
        h_gt_given_pred += (col_total / total) * float(-np.sum(probs * np.log(probs)))

    h_pred_given_gt = 0.0
    for row_idx, row_total in enumerate(row_sums):
        if row_total <= 0.0:
            continue
        probs = contingency[row_idx, :]
        probs = probs[probs > 0.0] / row_total
        h_pred_given_gt += (row_total / total) * float(-np.sum(probs * np.log(probs)))

    homogeneity = 1.0 if h_gt == 0.0 else 1.0 - (h_gt_given_pred / h_gt)
    completeness = 1.0 if h_pred == 0.0 else 1.0 - (h_pred_given_gt / h_pred)
    if homogeneity + completeness <= 1e-12:
        return 0.0
    return float((2.0 * homogeneity * completeness) / (homogeneity + completeness))


def _linear_sum_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if _scipy_linear_sum_assignment is not None:
        return _scipy_linear_sum_assignment(cost_matrix)

    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    if cost_matrix.ndim != 2:
        raise ValueError("cost_matrix must be 2-dimensional")
    n_rows, n_cols = cost_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    transposed = False
    if n_rows > n_cols:
        cost_matrix = cost_matrix.T
        n_rows, n_cols = cost_matrix.shape
        transposed = True

    u = np.zeros((n_rows + 1,), dtype=np.float64)
    v = np.zeros((n_cols + 1,), dtype=np.float64)
    p = np.zeros((n_cols + 1,), dtype=np.int64)
    way = np.zeros((n_cols + 1,), dtype=np.int64)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full((n_cols + 1,), np.inf, dtype=np.float64)
        used = np.zeros((n_cols + 1,), dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost_matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_ind = []
    col_ind = []
    for j in range(1, n_cols + 1):
        if p[j] != 0:
            row_ind.append(int(p[j] - 1))
            col_ind.append(int(j - 1))

    row_ind_arr = np.asarray(row_ind, dtype=np.int64)
    col_ind_arr = np.asarray(col_ind, dtype=np.int64)
    if transposed:
        return col_ind_arr, row_ind_arr
    return row_ind_arr, col_ind_arr


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

    row_ind, col_ind = _linear_sum_assignment(-overlap)
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
        "accuracy": _accuracy_score(gt_labels, pred_labels) if gt_labels else 0.0,
        "macro_f1": _macro_f1_score(gt_labels, pred_labels) if gt_labels else 0.0,
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
            "v_measure": _v_measure_score(gt_labels, pred_labels) if gt_labels else 0.0,
            "adjusted_rand_index": _adjusted_rand_score(gt_labels, pred_labels) if gt_labels else 0.0,
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
