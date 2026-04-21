"""Compatibility wrapper for the legacy stage-1 clustering pipeline.

The implementation now lives under the `clustering` package so the tracklet,
distance, and catalogue logic can be reused without importing one large file.
"""

from clustering.catalogue import assign_person_ids, build_catalogue
from clustering.common import (
    _build_detection_weights,
    _check_distance_matrix,
    _compute_motion_profile,
    _frame_ranges,
    _normalize_rows,
    _parse_clip_start_time,
    _safe_float,
    _smooth_embeddings,
    tracklets_cooccur,
)
from clustering.distance import (
    _apply_temporal_motion_penalty,
    _base_tracklet_distance,
    _tracklet_pair_distance,
    build_dense_candidate_pairs,
    build_distance_matrix,
    build_sparse_candidate_pairs,
)
from clustering.legacy import (
    _cluster_from_sorted_pairs,
    cluster_tracklets,
    cluster_tracklets_sparse,
    generate_person_catalogue,
    postprocess_singleton_merges,
)
from clustering.tracklets import build_tracklets, compute_tracklet_info, compute_tracklet_representative

__all__ = [
    "_apply_temporal_motion_penalty",
    "_base_tracklet_distance",
    "_build_detection_weights",
    "_check_distance_matrix",
    "_cluster_from_sorted_pairs",
    "_compute_motion_profile",
    "_frame_ranges",
    "_normalize_rows",
    "_parse_clip_start_time",
    "_safe_float",
    "_smooth_embeddings",
    "_tracklet_pair_distance",
    "assign_person_ids",
    "build_catalogue",
    "build_dense_candidate_pairs",
    "build_distance_matrix",
    "build_sparse_candidate_pairs",
    "build_tracklets",
    "cluster_tracklets",
    "cluster_tracklets_sparse",
    "compute_tracklet_info",
    "compute_tracklet_representative",
    "generate_person_catalogue",
    "postprocess_singleton_merges",
    "tracklets_cooccur",
]
