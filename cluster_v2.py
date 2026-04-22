from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from clustering.catalogue import assign_person_ids, build_catalogue
from clustering.common import _compute_motion_profile, _frame_ranges, _parse_clip_start_time, _safe_float, _smooth_embeddings, tracklets_cooccur
from clustering.tracklets import build_tracklets
from reid_ensemble import filter_crops_for_reid
from reid_model import torso_color_chi2
from rerank import kreciprocal_rerank

LOGGER = logging.getLogger(__name__)

try:
    from sklearn.metrics import pairwise_distances as _sklearn_pairwise_distances
    from sklearn.preprocessing import normalize as _sklearn_normalize
except ImportError:
    _sklearn_pairwise_distances = None
    _sklearn_normalize = None


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.float32, copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0.0, 1.0, norms)


def _normalize(arr: np.ndarray) -> np.ndarray:
    if _sklearn_normalize is not None:
        return _sklearn_normalize(arr)
    return _normalize_rows(arr)


def _pairwise_distances(arr: np.ndarray) -> np.ndarray:
    if _sklearn_pairwise_distances is not None:
        return _sklearn_pairwise_distances(arr, metric="cosine")
    normalized = _normalize_rows(arr)
    distances = 1.0 - np.matmul(normalized, normalized.T)
    return np.clip(distances, 0.0, 2.0).astype(np.float32, copy=False)


class _KMedoids:
    """Minimal PAM-style k-medoids matching the slice of sklearn_extra.cluster.KMedoids we use."""

    def __init__(self, n_clusters: int, metric: str = "cosine", random_state: int = 0, max_iter: int = 100):
        self.n_clusters = int(n_clusters)
        self.metric = metric
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.medoid_indices_: Optional[np.ndarray] = None

    def fit(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        k = max(1, min(self.n_clusters, n))
        dist = _pairwise_distances(X)

        rng = np.random.default_rng(self.random_state)
        medoids = [int(rng.integers(0, n))]
        while len(medoids) < k:
            min_d = np.maximum(dist[:, medoids].min(axis=1), 0.0)
            total = float(min_d.sum())
            if total <= 0.0:
                remaining = np.setdiff1d(np.arange(n), np.asarray(medoids), assume_unique=False)
                if remaining.size == 0:
                    break
                medoids.append(int(rng.choice(remaining)))
            else:
                medoids.append(int(rng.choice(n, p=min_d / total)))
        medoids_arr = np.asarray(medoids, dtype=np.int64)

        for _ in range(self.max_iter):
            labels = np.argmin(dist[:, medoids_arr], axis=1)
            new_medoids = medoids_arr.copy()
            for j in range(medoids_arr.size):
                members = np.where(labels == j)[0]
                if members.size == 0:
                    continue
                sub_cost = dist[np.ix_(members, members)].sum(axis=1)
                new_medoids[j] = members[int(np.argmin(sub_cost))]
            if np.array_equal(new_medoids, medoids_arr):
                break
            medoids_arr = new_medoids

        self.medoid_indices_ = medoids_arr
        return self


def _get_clustering_backends():
    import hdbscan

    return _KMedoids, hdbscan.HDBSCAN


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _detection_quality(det: dict) -> float:
    quality = _safe_float(det.get("quality"))
    if quality is not None:
        return quality
    confidence = _safe_float(det.get("confidence"))
    return confidence if confidence is not None else 0.0


def _load_rgb_frame(video_path: str, frame_num: int) -> Optional[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(frame_num) - 1, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _visible_pose_keypoints(frame_rgb: np.ndarray, box_xyxy: np.ndarray, pose_model) -> int:
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.int32)
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), frame_rgb.shape[1])
    y2 = min(int(y2), frame_rgb.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return 0
    result = pose_model.process(crop)
    landmarks = getattr(result, "pose_landmarks", None)
    if landmarks is None:
        return 0
    return sum(
        1
        for landmark in landmarks.landmark
        if getattr(landmark, "visibility", 0.0) > 0.5
    )


def _init_pose_model(use_pose: bool):
    if not use_pose:
        return None
    import mediapipe as mp

    return mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    )


def _select_tracklet_detections(
    dets_sorted: List[dict],
    frame_groups: Dict[Tuple[str, int], List[dict]],
    top_k_frames: int,
    use_pose: bool,
    pose_model,
) -> List[dict]:
    ranked = sorted(dets_sorted, key=_detection_quality, reverse=True)
    selected: List[dict] = []

    for det in ranked:
        box = np.asarray(det.get("box_xyxy_abs") or [], dtype=np.float32)
        frame_h = int(det.get("frame_height") or 0)
        frame_w = int(det.get("frame_width") or 0)
        if box.shape != (4,) or frame_h <= 0 or frame_w <= 0:
            selected.append(det)
        else:
            sibling_boxes = []
            for other in frame_groups.get((str(det["clip_id"]), int(det["frame_num"])), []):
                other_box = np.asarray(other.get("box_xyxy_abs") or [], dtype=np.float32)
                if other_box.shape == (4,):
                    sibling_boxes.append(other_box)
            sibling_boxes_arr = (
                np.stack(sibling_boxes, axis=0)
                if sibling_boxes
                else np.empty((0, 4), dtype=np.float32)
            )

            pose_callable = None
            if use_pose and pose_model is not None and det.get("video_path"):
                frame_rgb = _load_rgb_frame(det["video_path"], int(det["frame_num"]))
                if frame_rgb is not None:
                    pose_callable = lambda current_box, frame_rgb=frame_rgb: _visible_pose_keypoints(
                        frame_rgb,
                        current_box,
                        pose_model,
                    )

            keep = filter_crops_for_reid(
                boxes_xyxy=box[None, :],
                frame_shape=(frame_h, frame_w),
                other_boxes=sibling_boxes_arr,
                use_pose=use_pose,
                pose_model=pose_callable,
            )
            if bool(keep[0]):
                selected.append(det)

        if len(selected) >= top_k_frames:
            break

    if not selected:
        return ranked[:top_k_frames]
    return selected


def _compute_tracklet_profiles(
    tracklets: Dict[Tuple[str, int], List[dict]],
    frame_groups: Dict[Tuple[str, int], List[dict]],
    video_meta: dict,
    top_k_frames: int,
    n_prototypes: int,
    seed: int,
    use_pose: bool,
) -> List[dict]:
    KMedoids, _ = _get_clustering_backends()
    tracklet_info = []
    pose_model = _init_pose_model(use_pose)

    for (clip_id, track_id), dets in sorted(tracklets.items(), key=lambda item: item[0]):
        dets_sorted = sorted(dets, key=lambda det: int(det["frame_num"]))
        selected = _select_tracklet_detections(
            dets_sorted,
            frame_groups=frame_groups,
            top_k_frames=top_k_frames,
            use_pose=use_pose,
            pose_model=pose_model,
        )
        embeddings = np.asarray([det["embeddings"] for det in selected], dtype=np.float32)
        if embeddings.size == 0:
            continue
        embeddings = _normalize(embeddings.astype(np.float32))
        embeddings = _smooth_embeddings(embeddings, 5)

        if len(selected) < 6:
            prototypes = _normalize(embeddings.mean(axis=0, keepdims=True).astype(np.float32))
        else:
            proto_count = min(int(n_prototypes), len(selected))
            try:
                medoids = KMedoids(
                    n_clusters=proto_count,
                    metric="cosine",
                    random_state=seed,
                ).fit(embeddings)
                prototypes = embeddings[np.asarray(medoids.medoid_indices_, dtype=np.int64)]
            except Exception as exc:
                LOGGER.warning("KMedoids failed for %s_%s: %s", clip_id, track_id, exc)
                prototypes = _normalize(embeddings.mean(axis=0, keepdims=True).astype(np.float32))

        frames = [int(det["frame_num"]) for det in dets_sorted]
        ts_values = [
            _safe_float(det.get("timestamp_sec"))
            for det in dets_sorted
            if _safe_float(det.get("timestamp_sec")) is not None
        ]
        rel_center_sec = float(np.median(ts_values)) if ts_values else None
        clip_start_sec = _parse_clip_start_time(clip_id)
        abs_center_sec = (
            clip_start_sec + rel_center_sec
            if clip_start_sec is not None and rel_center_sec is not None
            else None
        )
        meta = video_meta.get(str(clip_id), {})
        frame_count = max(int(meta.get("frame_count") or max(frames)), 1)
        center_ratio = (frames[0] + frames[-1]) * 0.5 / frame_count

        torso_hists = [
            np.asarray(det["torso_hist"], dtype=np.float32)
            for det in selected
            if det.get("torso_hist") is not None
        ]
        torso_hist = None
        if torso_hists:
            torso_hist = _normalize(
                np.mean(np.stack(torso_hists, axis=0), axis=0, keepdims=True).astype(np.float32)
            )[0]

        tracklet_info.append(
            {
                "clip_id": str(clip_id),
                "track_id": int(track_id),
                "frame_ranges": _frame_ranges(frames),
                "num_frames": len(frames),
                "prototypes": _normalize(np.asarray(prototypes, dtype=np.float32)),
                "embedding": _normalize(np.asarray(prototypes, dtype=np.float32)).mean(axis=0),
                "all_embeddings": embeddings,
                "relative_time_center_sec": rel_center_sec,
                "absolute_time_center_sec": abs_center_sec,
                "temporal_center_ratio": center_ratio,
                "motion_speed": _compute_motion_profile(dets_sorted),
                "torso_hist": torso_hist,
                "quality_mean": float(np.mean([_detection_quality(det) for det in selected])),
            }
        )

    if pose_model is not None:
        pose_model.close()
    return tracklet_info


def _build_tracklet_distance(
    tracklet_info: List[dict],
    use_rerank: bool,
    rerank_k1: int,
    rerank_k2: int,
    rerank_lambda: float,
    cooccurrence_constraint: bool,
) -> np.ndarray:
    proto_feats = []
    proto_owner = []
    for idx, tracklet in enumerate(tracklet_info):
        for prototype in tracklet["prototypes"]:
            proto_feats.append(prototype)
            proto_owner.append(idx)

    proto_feats = _normalize(np.asarray(proto_feats, dtype=np.float32))
    proto_owner = np.asarray(proto_owner, dtype=np.int32)
    if use_rerank:
        d_proto = kreciprocal_rerank(proto_feats, rerank_k1, rerank_k2, rerank_lambda)
    else:
        d_proto = _pairwise_distances(proto_feats).astype(np.float32)

    n_tracklets = len(tracklet_info)
    d_track = np.full((n_tracklets, n_tracklets), np.inf, dtype=np.float32)
    np.fill_diagonal(d_track, 0.0)

    for i in range(n_tracklets):
        for j in range(i + 1, n_tracklets):
            mask_i = proto_owner == i
            mask_j = proto_owner == j
            d_track[i, j] = float(np.min(d_proto[np.ix_(mask_i, mask_j)]))
            d_track[j, i] = d_track[i, j]
            if cooccurrence_constraint and tracklets_cooccur(tracklet_info[i], tracklet_info[j]):
                d_track[i, j] = np.inf
                d_track[j, i] = np.inf

    return d_track


def _cluster_hist(tracklet_info: List[dict], members: np.ndarray) -> Optional[np.ndarray]:
    hists = [
        tracklet_info[int(member)]["torso_hist"]
        for member in members
        if tracklet_info[int(member)]["torso_hist"] is not None
    ]
    if not hists:
        return None
    return _normalize(np.mean(np.stack(hists, axis=0), axis=0, keepdims=True))[0]


def generate_person_catalogue_v2(
    all_detections,
    video_meta: dict,
    output_file: str = "catalogue_v2.json",
    top_k_frames: int = 20,
    n_prototypes: int = 3,
    use_rerank: bool = True,
    rerank_k1: int = 20,
    rerank_k2: int = 6,
    rerank_lambda: float = 0.3,
    cooccurrence_constraint: bool = True,
    min_cluster_size: int = 2,
    color_tiebreak: bool = True,
    seed: int = 51,
    use_pose: bool = False,
) -> dict:
    if not all_detections:
        LOGGER.warning("No detections found for V2 clustering")
        return {}

    tracklets = build_tracklets(all_detections)
    frame_groups: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for det in all_detections:
        frame_groups[(str(det["clip_id"]), int(det["frame_num"]))].append(det)

    tracklet_info = _compute_tracklet_profiles(
        tracklets=tracklets,
        frame_groups=frame_groups,
        video_meta=video_meta,
        top_k_frames=top_k_frames,
        n_prototypes=n_prototypes,
        seed=seed,
        use_pose=use_pose,
    )
    if not tracklet_info:
        LOGGER.warning("No tracklet profiles produced for V2 clustering")
        return {}

    d_track = _build_tracklet_distance(
        tracklet_info=tracklet_info,
        use_rerank=use_rerank,
        rerank_k1=rerank_k1,
        rerank_k2=rerank_k2,
        rerank_lambda=rerank_lambda,
        cooccurrence_constraint=cooccurrence_constraint,
    )
    valid = d_track[np.isfinite(d_track) & (d_track > 0)]
    epsilon = max(float(np.percentile(valid, 5)), 0.18) if valid.size else 0.18
    gate = float(np.percentile(valid, 15)) if valid.size else epsilon
    LOGGER.info("Adaptive clustering epsilon=%.4f gate=%.4f", epsilon, gate)

    _, HDBSCAN = _get_clustering_backends()
    clusterer = HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=1,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method="eom",
    ).fit(np.ascontiguousarray(d_track, dtype=np.float64))
    labels = np.asarray(clusterer.labels_, dtype=np.int32)
    probabilities = np.asarray(
        getattr(clusterer, "probabilities_", np.ones_like(labels, dtype=np.float32)),
        dtype=np.float32,
    )

    cluster_ids = sorted(int(label) for label in np.unique(labels) if label != -1)
    for idx in np.where(labels == -1)[0]:
        best_cluster = None
        best_dist = float("inf")
        best_chi2 = 0.0
        for cluster_id in cluster_ids:
            members = np.where(labels == cluster_id)[0]
            if members.size == 0:
                continue
            avg_dist = float(np.mean(d_track[idx, members]))
            if avg_dist < best_dist:
                best_dist = avg_dist
                best_cluster = cluster_id
                cluster_hist = _cluster_hist(tracklet_info, members)
                if color_tiebreak and cluster_hist is not None and tracklet_info[idx]["torso_hist"] is not None:
                    best_chi2 = torso_color_chi2(tracklet_info[idx]["torso_hist"], cluster_hist)
                else:
                    best_chi2 = 0.0

        accept = (
            best_cluster is not None
            and np.isfinite(best_dist)
            and best_dist <= gate
            and (not color_tiebreak or best_chi2 <= 0.5)
        )
        if accept:
            LOGGER.info(
                "merge %s_%s: dist=%.2f, chi2=%.2f -> accept",
                tracklet_info[idx]["clip_id"],
                tracklet_info[idx]["track_id"],
                best_dist,
                best_chi2,
            )
            labels[idx] = int(best_cluster)
            probabilities[idx] = max(probabilities[idx], 0.5)

    for idx, tracklet in enumerate(tracklet_info):
        tracklet["cluster_probability"] = float(probabilities[idx])

    assign_person_ids(tracklet_info, labels)
    catalogue = build_catalogue(tracklet_info)

    for tracklet in tracklet_info:
        for appearance in catalogue[str(tracklet["global_id"])]:
            if (
                appearance["clip_id"] == tracklet["clip_id"]
                and appearance["local_track_id"] == tracklet["track_id"]
            ):
                appearance["cluster_probability"] = tracklet["cluster_probability"]
                break

    summary = {
        "total_unique_persons": len(catalogue),
        "total_tracklets": len(tracklet_info),
        "parameters": {
            "top_k_frames": top_k_frames,
            "n_prototypes": n_prototypes,
            "use_rerank": use_rerank,
            "rerank_k1": rerank_k1,
            "rerank_k2": rerank_k2,
            "rerank_lambda": rerank_lambda,
            "cooccurrence_constraint": cooccurrence_constraint,
            "min_cluster_size": min_cluster_size,
            "color_tiebreak": color_tiebreak,
            "seed": seed,
            "use_pose": use_pose,
        },
        "adaptive": {
            "epsilon": epsilon,
            "gate": gate,
        },
    }
    output = {"summary": summary, "catalogue": catalogue}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    LOGGER.info("Catalogue V2 saved to %s", Path(output_file).resolve())
    return output
