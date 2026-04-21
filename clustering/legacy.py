from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from clustering.catalogue import assign_person_ids, build_catalogue
from clustering.common import _components_cooccur, tracklets_cooccur
from clustering.distance import (
    _tracklet_pair_distance,
    build_dense_candidate_pairs,
    build_distance_matrix,
    build_sparse_candidate_pairs,
)
from clustering.tracklets import build_tracklets, compute_tracklet_info


def _cluster_from_sorted_pairs(
        sorted_pairs: List[Tuple[float, int, int]],
        tracklet_info: List[Dict[str, Any]],
        min_cluster_size: int,
        cluster_selection_epsilon: float,
        cooccurrence_constraint: bool = True,
) -> np.ndarray:
    """Union-find clustering with cross-clip or co-occurrence constraints."""
    n = len(tracklet_info)

    parent = list(range(n))
    rank = [0] * n
    clip_ids = [str(t["clip_id"]) for t in tracklet_info]
    comp_clip_sets: List[set] = [set([clip_ids[idx]]) for idx in range(n)]
    comp_members: List[List[int]] = [[idx] for idx in range(n)]

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px = find(x)
        py = find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            parent[px] = py
            comp_clip_sets[py] |= comp_clip_sets[px]
            comp_members[py].extend(comp_members[px])
        elif rank[px] > rank[py]:
            parent[py] = px
            comp_clip_sets[px] |= comp_clip_sets[py]
            comp_members[px].extend(comp_members[py])
        else:
            parent[py] = px
            comp_clip_sets[px] |= comp_clip_sets[py]
            comp_members[px].extend(comp_members[py])
            rank[px] += 1
        return True

    for dist, a, b in sorted_pairs:
        if dist > cluster_selection_epsilon:
            break
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if cooccurrence_constraint:
            if _components_cooccur(comp_members[ra], comp_members[rb], tracklet_info):
                continue
        elif not comp_clip_sets[ra].isdisjoint(comp_clip_sets[rb]):
            continue
        union(ra, rb)

    components = defaultdict(list)
    for idx in range(n):
        root = find(idx)
        components[root].append(idx)

    cluster_groups = {k: v for k, v in components.items() if len(v) >= min_cluster_size}
    labels = np.full(n, -1, dtype=np.int32)
    for cl_id, members in enumerate(cluster_groups.values()):
        for m in members:
            labels[m] = cl_id

    print(f"Raw labels: {list(labels)}")

    n_clusters = len(cluster_groups)
    n_noise = n - sum(len(v) for v in cluster_groups.values())
    total_persons = n_clusters + n_noise

    print(
        f"Custom clustering mcs={min_cluster_size}, eps={cluster_selection_epsilon:.3f} -> "
        f"clusters={n_clusters}, noise={n_noise}, total_persons={total_persons}"
    )

    print("\nClusters:")
    for cluster_id, indices in enumerate(cluster_groups.values()):
        members = [f"{tracklet_info[i]['clip_id']}_{tracklet_info[i]['track_id']}" for i in sorted(indices)]
        print(f"  Cluster {cluster_id}: {{{', '.join(members)}}}")

    noise_indices = np.where(labels == -1)[0]
    if len(noise_indices) > 0:
        noise_members = [f"{tracklet_info[i]['clip_id']}_{tracklet_info[i]['track_id']}" for i in sorted(noise_indices)]
        print(f"  Noise: {{{', '.join(noise_members)}}}")

    return labels


def cluster_tracklets(
        dist_matrix: np.ndarray,
        tracklet_info: List[Dict[str, Any]],
        min_cluster_size: int = 2,
        cluster_selection_epsilon: float = 0.025,
        cooccurrence_constraint: bool = True,
) -> np.ndarray:
    """Cluster tracklets using custom greedy clustering."""
    print(
        f"Using custom greedy clustering, epsilon={cluster_selection_epsilon}, "
        f"cooccurrence_constraint={cooccurrence_constraint}"
    )

    sorted_pairs, total_pairs, threshold = build_dense_candidate_pairs(
        dist_matrix, cluster_selection_epsilon
    )
    print(
        f"Pair pre-filter: kept {len(sorted_pairs):,}/{total_pairs:,} "
        f"({(100.0 * len(sorted_pairs) / total_pairs) if total_pairs else 0.0:.1f}%) "
        f"with dist <= {threshold:.4f}"
    )
    return _cluster_from_sorted_pairs(
        sorted_pairs=sorted_pairs,
        tracklet_info=tracklet_info,
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cooccurrence_constraint=cooccurrence_constraint,
    )


def cluster_tracklets_sparse(
        tracklet_info: List[Dict[str, Any]],
        min_cluster_size: int = 2,
        cluster_selection_epsilon: float = 0.025,
        linkage: str = "min",
        temporal_penalty: float = 0.0,
        temporal_max_gap_sec: Optional[float] = None,
        motion_penalty: float = 0.0,
        cooccurrence_constraint: bool = True,
) -> np.ndarray:
    """Sparse radius-neighbor variant for large tracklet counts."""
    print(
        f"Using sparse radius-neighbor clustering, epsilon={cluster_selection_epsilon}, "
        f"linkage={linkage}, cooccurrence_constraint={cooccurrence_constraint}"
    )
    sorted_pairs, total_pairs, threshold = build_sparse_candidate_pairs(
        tracklet_info,
        cluster_selection_epsilon,
        linkage=linkage,
        temporal_penalty=temporal_penalty,
        temporal_max_gap_sec=temporal_max_gap_sec,
        motion_penalty=motion_penalty,
        cooccurrence_constraint=cooccurrence_constraint,
    )
    print(
        f"Sparse candidate graph: kept {len(sorted_pairs):,}/{total_pairs:,} "
        f"({(100.0 * len(sorted_pairs) / total_pairs) if total_pairs else 0.0:.1f}%) "
        f"with dist <= {threshold:.4f}"
    )
    return _cluster_from_sorted_pairs(
        sorted_pairs=sorted_pairs,
        tracklet_info=tracklet_info,
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cooccurrence_constraint=cooccurrence_constraint,
    )


def postprocess_singleton_merges(
        labels: np.ndarray,
        tracklet_info: List[Dict[str, Any]],
        merge_epsilon: float,
        linkage: str = "min",
        temporal_penalty: float = 0.0,
        temporal_max_gap_sec: Optional[float] = None,
        motion_penalty: float = 0.0,
        cooccurrence_constraint: bool = True,
) -> np.ndarray:
    """Post-process noise labels by merging singleton tracklets into nearest valid cluster."""
    if labels.size == 0:
        return labels

    labels = labels.copy()
    singleton_indices = np.where(labels == -1)[0]
    if len(singleton_indices) == 0:
        return labels

    cluster_ids = sorted(int(c) for c in np.unique(labels) if c != -1)
    if not cluster_ids:
        return labels

    merged = 0
    for idx in singleton_indices:
        clip_id = tracklet_info[idx]["clip_id"]
        best_cluster = None
        best_dist = float("inf")

        for cluster_id in cluster_ids:
            members = np.where(labels == cluster_id)[0]
            if cooccurrence_constraint:
                if any(
                    tracklets_cooccur(tracklet_info[idx], tracklet_info[m])
                    for m in members
                ):
                    continue
            elif any(tracklet_info[m]["clip_id"] == clip_id for m in members):
                continue
            d = min(
                _tracklet_pair_distance(
                    tracklet_info[idx],
                    tracklet_info[m],
                    linkage=linkage,
                    temporal_penalty=temporal_penalty,
                    temporal_max_gap_sec=temporal_max_gap_sec,
                    motion_penalty=motion_penalty,
                )
                for m in members
            )
            if d < best_dist:
                best_dist = d
                best_cluster = cluster_id

        if best_cluster is not None and np.isfinite(best_dist) and best_dist <= merge_epsilon:
            labels[idx] = best_cluster
            merged += 1

    if merged:
        print(
            f"Post-processing: merged {merged} singleton tracklets into existing clusters "
            f"(merge_epsilon={merge_epsilon:.3f})."
        )
    return labels


def generate_person_catalogue(
        all_detections: List[Dict[str, Any]],
        output_file: str = "catalogue_simple.json",
        min_cluster_size: int = 2,
        cluster_selection_epsilon: float = 0.35,
        use_median: bool = True,
        print_distances: bool = False,
        use_sparse_neighbors: bool = True,
        sparse_if_n_ge: int = 1000,
        linkage: str = "min",
        min_tracklet_frames: int = 2,
        use_quality_weights: bool = True,
        quality_alpha: float = 0.75,
        smooth_embeddings: bool = True,
        smoothing_window: int = 5,
        temporal_penalty: float = 0.05,
        temporal_max_gap_sec: Optional[float] = None,
        motion_penalty: float = 0.05,
        postprocess_merge: bool = True,
        postprocess_merge_epsilon: Optional[float] = None,
        use_rerank: bool = False,
        cooccurrence_constraint: bool = True,
):
    """Legacy stage-1 clustering pipeline."""
    if not all_detections:
        print("No detections found.")
        return {}

    quality_alpha = float(np.clip(quality_alpha, 0.0, 1.0))
    temporal_penalty = max(0.0, float(temporal_penalty))
    motion_penalty = max(0.0, float(motion_penalty))

    tracklets = build_tracklets(all_detections)
    print(f"Found {len(tracklets)} tracklets")

    tracklet_info = compute_tracklet_info(
        tracklets,
        use_median=use_median,
        min_tracklet_frames=min_tracklet_frames,
        use_quality_weights=use_quality_weights,
        quality_alpha=quality_alpha,
        smooth_embeddings=smooth_embeddings,
        smoothing_window=smoothing_window,
    )
    if not tracklet_info:
        print("No tracklet info produced.")
        return {}

    n_tracklets = len(tracklet_info)
    use_sparse_path = (
        use_sparse_neighbors
        and n_tracklets >= sparse_if_n_ge
        and not print_distances
        and not use_rerank
    )
    if use_sparse_path:
        labels = cluster_tracklets_sparse(
            tracklet_info,
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
            linkage=linkage,
            temporal_penalty=temporal_penalty,
            temporal_max_gap_sec=temporal_max_gap_sec,
            motion_penalty=motion_penalty,
            cooccurrence_constraint=cooccurrence_constraint,
        )
    else:
        dist_matrix = build_distance_matrix(
            tracklet_info,
            print_distances=print_distances,
            linkage=linkage,
            temporal_penalty=temporal_penalty,
            temporal_max_gap_sec=temporal_max_gap_sec,
            motion_penalty=motion_penalty,
            use_rerank=use_rerank,
        )
        if print_distances:
            print(f"dist_matrix: {dist_matrix}")
        labels = cluster_tracklets(
            dist_matrix,
            tracklet_info,
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cooccurrence_constraint=cooccurrence_constraint,
        )

    merge_eps_used = None
    if postprocess_merge:
        merge_eps_used = (
            float(postprocess_merge_epsilon)
            if postprocess_merge_epsilon is not None
            else float(cluster_selection_epsilon) * 0.65
        )
        labels = postprocess_singleton_merges(
            labels=labels,
            tracklet_info=tracklet_info,
            merge_epsilon=merge_eps_used,
            linkage=linkage,
            temporal_penalty=temporal_penalty,
            temporal_max_gap_sec=temporal_max_gap_sec,
            motion_penalty=motion_penalty,
            cooccurrence_constraint=cooccurrence_constraint,
        )

    assign_person_ids(tracklet_info, labels)
    catalogue = build_catalogue(tracklet_info)

    summary = {
        "total_unique_persons": len(catalogue),
        "total_tracklets": len(tracklet_info),
        "parameters": {
            "min_cluster_size": min_cluster_size,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "use_median": use_median,
            "cross_clip_only": not cooccurrence_constraint,
            "cooccurrence_constraint": cooccurrence_constraint,
            "linkage": linkage,
            "min_tracklet_frames": min_tracklet_frames,
            "sparse_neighbors_enabled": use_sparse_neighbors,
            "sparse_if_n_ge": sparse_if_n_ge,
            "sparse_path_used": use_sparse_path,
            "use_rerank": use_rerank,
            "use_quality_weights": use_quality_weights,
            "quality_alpha": quality_alpha,
            "smooth_embeddings": smooth_embeddings,
            "smoothing_window": smoothing_window,
            "temporal_penalty": temporal_penalty,
            "temporal_max_gap_sec": temporal_max_gap_sec,
            "motion_penalty": motion_penalty,
            "postprocess_merge": postprocess_merge,
            "postprocess_merge_epsilon": merge_eps_used,
        },
    }
    output = {"summary": summary, "catalogue": catalogue}

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Unique persons: {summary['total_unique_persons']}")
    print(f"Tracklets clustered: {summary['total_tracklets']}")
    print(f"Catalogue saved to: {Path(output_file).resolve()}")

    return catalogue
