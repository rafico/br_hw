from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from clustering.common import _check_distance_matrix, tracklets_cooccur
from rerank import kreciprocal_rerank


def _base_tracklet_distance(
        t1: Dict[str, Any],
        t2: Dict[str, Any],
        linkage: str,
) -> float:
    """Base embedding distance before temporal/motion penalties."""
    if linkage == "representative":
        return float(pairwise_distances(
            t1["embedding"][None, :],
            t2["embedding"][None, :],
            metric="cosine",
        )[0, 0])

    D = pairwise_distances(
        t1["all_embeddings"],
        t2["all_embeddings"],
        metric="cosine",
    )
    if linkage == "min":
        return float(D.min())
    return float(D.mean())


def _apply_temporal_motion_penalty(
        base_distance: float,
        t1: Dict[str, Any],
        t2: Dict[str, Any],
        temporal_penalty: float = 0.0,
        temporal_max_gap_sec: Optional[float] = None,
        motion_penalty: float = 0.0,
) -> float:
    """Adjust distance using optional temporal and motion consistency penalties."""
    dist = float(base_distance)

    if temporal_penalty > 0:
        c1 = t1.get("temporal_center_ratio")
        c2 = t2.get("temporal_center_ratio")
        if c1 is not None and c2 is not None:
            dist += float(temporal_penalty) * abs(float(c1) - float(c2))

    if temporal_max_gap_sec is not None:
        a1 = t1.get("absolute_time_center_sec")
        a2 = t2.get("absolute_time_center_sec")
        if a1 is not None and a2 is not None:
            if abs(float(a1) - float(a2)) > float(temporal_max_gap_sec):
                return np.inf

    if motion_penalty > 0:
        m1 = t1.get("motion_speed")
        m2 = t2.get("motion_speed")
        if m1 is not None and m2 is not None:
            dist += float(motion_penalty) * abs(float(m1) - float(m2))

    return dist


def _tracklet_pair_distance(
        t1: Dict[str, Any],
        t2: Dict[str, Any],
        linkage: str = "min",
        temporal_penalty: float = 0.0,
        temporal_max_gap_sec: Optional[float] = None,
        motion_penalty: float = 0.0,
) -> float:
    base = _base_tracklet_distance(t1, t2, linkage=linkage)
    return _apply_temporal_motion_penalty(
        base,
        t1,
        t2,
        temporal_penalty=temporal_penalty,
        temporal_max_gap_sec=temporal_max_gap_sec,
        motion_penalty=motion_penalty,
    )


def build_distance_matrix(
        tracklet_info: List[Dict[str, Any]],
        print_distances: bool = False,
        linkage: str = "min",
        temporal_penalty: float = 0.0,
        temporal_max_gap_sec: Optional[float] = None,
        motion_penalty: float = 0.0,
        use_rerank: bool = False,
) -> np.ndarray:
    """Build pairwise distance matrix between tracklets."""
    n = len(tracklet_info)
    dist_matrix = np.zeros((n, n), dtype=np.float32)

    if print_distances:
        print("Tracklet order:")
        for i, t in enumerate(tracklet_info):
            print(f"  {i}: {t['clip_id']}_{t['track_id']} ({t['num_frames']} frames)")

    if use_rerank:
        X = np.stack([t["embedding"] for t in tracklet_info], axis=0).astype(np.float32)
        X = normalize(X)
        base = kreciprocal_rerank(X)
        for i in range(n):
            for j in range(i + 1, n):
                d = _apply_temporal_motion_penalty(
                    float(base[i, j]),
                    tracklet_info[i],
                    tracklet_info[j],
                    temporal_penalty=temporal_penalty,
                    temporal_max_gap_sec=temporal_max_gap_sec,
                    motion_penalty=motion_penalty,
                )
                if not np.isfinite(d):
                    d = 2.0
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
    elif linkage == "representative":
        X = np.stack([t["embedding"] for t in tracklet_info], axis=0).astype(np.float32)
        X = normalize(X)
        base = pairwise_distances(X, metric="cosine")
        for i in range(n):
            for j in range(i + 1, n):
                d = _apply_temporal_motion_penalty(
                    float(base[i, j]),
                    tracklet_info[i],
                    tracklet_info[j],
                    temporal_penalty=temporal_penalty,
                    temporal_max_gap_sec=temporal_max_gap_sec,
                    motion_penalty=motion_penalty,
                )
                if not np.isfinite(d):
                    d = 2.0
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
    else:
        for i in range(n):
            for j in range(i + 1, n):
                d = _tracklet_pair_distance(
                    tracklet_info[i],
                    tracklet_info[j],
                    linkage=linkage,
                    temporal_penalty=temporal_penalty,
                    temporal_max_gap_sec=temporal_max_gap_sec,
                    motion_penalty=motion_penalty,
                )
                if not np.isfinite(d):
                    d = 2.0
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, 2.0)
    dist_matrix = np.ascontiguousarray(dist_matrix, dtype=np.float32)

    if print_distances:
        print("Pairwise distances between tracklets:")
        print(dist_matrix)

    _check_distance_matrix(dist_matrix)
    return dist_matrix


def build_dense_candidate_pairs(
        dist_matrix: np.ndarray,
        cluster_selection_epsilon: float,
) -> Tuple[List[Tuple[float, int, int]], int, float]:
    """Build sorted candidate pairs from a dense distance matrix."""
    i, j = np.triu_indices(dist_matrix.shape[0], k=1)
    dists = dist_matrix[i, j]
    threshold = cluster_selection_epsilon * 1.1
    keep = dists <= threshold
    kept_idx = np.where(keep)[0]
    pairs = [(float(dists[k]), int(i[k]), int(j[k])) for k in kept_idx]
    pairs.sort()
    return pairs, int(len(dists)), threshold


def build_sparse_candidate_pairs(
        tracklet_info: List[Dict[str, Any]],
        cluster_selection_epsilon: float,
        linkage: str = "min",
        temporal_penalty: float = 0.0,
        temporal_max_gap_sec: Optional[float] = None,
        motion_penalty: float = 0.0,
        cooccurrence_constraint: bool = False,
) -> Tuple[List[Tuple[float, int, int]], int, float]:
    """Build candidate pairs via sparse neighbor search."""
    n = len(tracklet_info)
    if n < 2:
        return [], 0, cluster_selection_epsilon * 1.1

    threshold = cluster_selection_epsilon * 1.1
    total_possible = (n * (n - 1)) // 2

    if linkage == "min":
        emb_chunks = []
        owners = []
        for idx, t in enumerate(tracklet_info):
            emb = np.asarray(t["all_embeddings"], dtype=np.float32)
            if emb.size == 0:
                continue
            emb_chunks.append(emb)
            owners.extend([idx] * len(emb))

        if not emb_chunks:
            return [], total_possible, threshold

        X = np.concatenate(emb_chunks, axis=0).astype(np.float32)
        owners_arr = np.asarray(owners, dtype=np.int32)
        nn = NearestNeighbors(metric="cosine", algorithm="brute", radius=threshold)
        nn.fit(X)
        distances, neighbors = nn.radius_neighbors(X, return_distance=True)

        best_pair_dist: Dict[Tuple[int, int], float] = {}
        for src_emb_idx, (dists, nbrs) in enumerate(zip(distances, neighbors)):
            src_t = int(owners_arr[src_emb_idx])
            for dist, dst_emb_idx in zip(dists, nbrs):
                dst_emb_idx = int(dst_emb_idx)
                if dst_emb_idx <= src_emb_idx:
                    continue
                dst_t = int(owners_arr[dst_emb_idx])
                if dst_t == src_t:
                    continue
                i, j = (src_t, dst_t) if src_t < dst_t else (dst_t, src_t)
                if (
                    not cooccurrence_constraint
                    and tracklet_info[i]["clip_id"] == tracklet_info[j]["clip_id"]
                ):
                    continue
                if tracklets_cooccur(tracklet_info[i], tracklet_info[j]):
                    continue

                d = _apply_temporal_motion_penalty(
                    float(dist),
                    tracklet_info[i],
                    tracklet_info[j],
                    temporal_penalty=temporal_penalty,
                    temporal_max_gap_sec=temporal_max_gap_sec,
                    motion_penalty=motion_penalty,
                )
                if not np.isfinite(d) or d > threshold:
                    continue

                key = (i, j)
                prev = best_pair_dist.get(key)
                if prev is None or d < prev:
                    best_pair_dist[key] = d

        pairs = [(d, i, j) for (i, j), d in best_pair_dist.items()]
        pairs.sort()
        return pairs, total_possible, threshold

    X = np.stack([t["embedding"] for t in tracklet_info], axis=0).astype(np.float32)
    X = normalize(X)
    search_radius = cluster_selection_epsilon * (1.5 if linkage != "representative" else 1.1)
    nn = NearestNeighbors(metric="cosine", algorithm="brute", radius=search_radius)
    nn.fit(X)
    distances, neighbors = nn.radius_neighbors(X, return_distance=True)

    pairs: List[Tuple[float, int, int]] = []
    for src, (dists, nbrs) in enumerate(zip(distances, neighbors)):
        for _, dst in zip(dists, nbrs):
            dst_i = int(dst)
            if dst_i <= src:
                continue
            if (
                not cooccurrence_constraint
                and tracklet_info[src]["clip_id"] == tracklet_info[dst_i]["clip_id"]
            ):
                continue
            if tracklets_cooccur(tracklet_info[src], tracklet_info[dst_i]):
                continue
            d = _tracklet_pair_distance(
                tracklet_info[src],
                tracklet_info[dst_i],
                linkage=linkage,
                temporal_penalty=temporal_penalty,
                temporal_max_gap_sec=temporal_max_gap_sec,
                motion_penalty=motion_penalty,
            )
            if np.isfinite(d) and d <= threshold:
                pairs.append((float(d), int(src), dst_i))

    pairs.sort()
    return pairs, total_possible, threshold
