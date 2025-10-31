import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import hdbscan  # ensure installed


# ----------------------------
# Helpers
# ----------------------------

def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def compute_tracklet_representative(embeddings: np.ndarray, use_median: bool = False) -> np.ndarray:
    """
    Simple representative embedding: L2-normalize each, take mean or median, re-normalize.
    Returns a 1D float64 vector.

    Args:
        embeddings: Array of embeddings for a tracklet
        use_median: If True, use median instead of mean for aggregation
    """
    if len(embeddings) == 0:
        raise ValueError("Tracklet has no embeddings")
    embeds = _normalize_rows(np.asarray(embeddings, dtype=np.float64))
    if use_median:
        emb = np.median(embeds, axis=0)
    else:
        emb = embeds.mean(axis=0)
    n = np.linalg.norm(emb)
    return emb if n == 0 else emb / n


def _frame_ranges(frame_list: List[int]) -> List[List[int]]:
    """
    Convert sorted list of frame numbers into [start, end] ranges.
    """
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


def _build_video_paths_from_dataset(dataset) -> Dict[str, str]:
    """
    Build {clip_id: filepath} from a FiftyOne dataset (clip_id = stem of filepath).
    """
    video_paths = {}
    for sample in dataset:
        clip_id = Path(sample.filepath).stem
        video_paths[clip_id] = sample.filepath
    return video_paths


def _check_distance_matrix(D: np.ndarray):
    """
    Basic sanity checks to catch issues early.
    """
    if not np.isfinite(D).all():
        raise ValueError("Distance matrix contains non-finite values.")
    if (D < 0).any():
        raise ValueError("Distance matrix has negative entries.")
    if not np.allclose(D, D.T, atol=1e-8):
        raise ValueError("Distance matrix must be symmetric.")


# ----------------------------
# Main pipeline
# ----------------------------

def generate_person_catalogue_and_save_clips(
        all_detections: List[Dict[str, Any]],
        dataset: Optional[Any] = None,
        video_paths: Optional[Dict[str, str]] = None,
        output_dir: str = "persons",
        output_file: str = "catalogue_simple.json",
        # Clustering controls
        target_clusters: int = 7,  # heuristic used to pick HDBSCAN params
        hdbscan_grid: Optional[Dict[str, Any]] = None,  # dict of lists
        # Representative embedding method
        use_median: bool = False,  # If True, use median instead of mean for tracklet representatives
        # Logging
        print_distances: bool = False,
        print_order: bool = True,
):
    """
    Minimal pipeline:
      1) Group detections by (clip_id, track_id) => tracklets
      2) Compute one representative embedding per tracklet (no outlier removal)
      3) Build cosine distance matrix (float64, contiguous)
      4) Cluster tracklets across clips (HDBSCAN grid sweep), preferring exactly `target_clusters`
      5) For each person, copy *entire* clip files they appear in into output_dir/person_XXX/

    Params:
      - all_detections: list of dicts with keys: clip_id, track_id, frame_num, embeddings
      - dataset: optional FiftyOne dataset, used to resolve clip filepaths
      - video_paths: optional dict {clip_id: absolute_filepath}; used if dataset is None
      - output_dir: root directory to place person folders
      - output_file: JSON path with a compact catalogue
      - target_clusters: desired number of clusters; used by the selection logic
      - hdbscan_grid: dict with keys "min_cluster_size", "min_samples", "cluster_selection_epsilon", "cluster_selection_method"
      - use_median: if True, use median instead of mean for computing tracklet representatives
      - print_distances: if True, pretty-print the distance matrix
      - print_order: if True, print tracklet order

    Returns:
      - catalogue: dict[str, list[appearance dict]]
    """
    if not all_detections:
        print("No detections found.")
        return {}

    # Resolve video paths
    if video_paths is None:
        if dataset is None:
            raise ValueError("Provide either `dataset` or `video_paths` to locate clip files.")
        video_paths = _build_video_paths_from_dataset(dataset)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1) Tracklets
    tracklets = defaultdict(list)
    for d in all_detections:
        tracklets[(str(d["clip_id"]), int(d["track_id"]))] += [d]
    print(f"Found {len(tracklets)} tracklets")

    # 2) Representatives
    tracklet_info = []
    for (clip_id, track_id), dets in tracklets.items():
        embeds = np.array([det["embeddings"] for det in dets], dtype=np.float64)
        rep = compute_tracklet_representative(embeds, use_median=use_median)
        frames = sorted([int(det["frame_num"]) for det in dets])
        tracklet_info.append(
            {
                "clip_id": clip_id,
                "track_id": track_id,
                "embedding": rep,
                "frame_ranges": _frame_ranges(frames),
                "num_frames": len(frames),
            }
        )

    if not tracklet_info:
        print("No tracklet info produced.")
        return {}

    # 3) Cosine distance matrix (float64, contiguous) + normalized X
    X = np.stack([t["embedding"] for t in tracklet_info], axis=0).astype(np.float64)
    X = normalize(X)  # keep unit length; HDBSCAN prefers consistent scaling

    if print_order:
        print("Tracklet order for distance matrix:")
        for i, t in enumerate(tracklet_info):
            print(f"{i}: {t['clip_id']}_{t['track_id']}")

    # Use sklearn to build float64 distances (for printing/sanity, even if we fit on X)
    dist_matrix = pairwise_distances(X, metric="cosine")  # returns float64
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, 2.0)
    dist_matrix = np.ascontiguousarray(dist_matrix, dtype=np.float64)

    if print_distances:
        np.set_printoptions(precision=6, suppress=True, linewidth=140)
        print("Pairwise cosine distances between tracklet representatives:")
        print(dist_matrix)

    _check_distance_matrix(dist_matrix)

    # ----------------------------
    # 4) Clustering with HDBSCAN
    # ----------------------------
    # More permissive default grid to make 7 clusters reachable
    if hdbscan_grid is None:
        hdbscan_grid = {
            "min_cluster_size": [2, 3, 4],
            "min_samples": [1, 2, 3],
            "cluster_selection_epsilon": [0.0, 0.005, 0.01, 0.02],
            "cluster_selection_method": "leaf",  # finer clusters than "eom"
        }

    mcs_list = hdbscan_grid["min_cluster_size"]
    ms_list = hdbscan_grid["min_samples"]
    eps_list = hdbscan_grid["cluster_selection_epsilon"]
    csm = hdbscan_grid.get("cluster_selection_method", "leaf")

    best = None
    best_labels = None
    best_clusterer = None

    def score_tuple(n_clusters: int, n_noise: int):
        # Prefer exact target first; then fewer noise points
        return (abs(n_clusters - target_clusters), n_noise)

    for mcs in mcs_list:
        hit_target = False
        for ms in ms_list:
            for cseps in eps_list:
                clusterer = hdbscan.HDBSCAN(
                    metric="precomputed",
                    min_cluster_size=mcs,
                    min_samples=ms,
                    cluster_selection_method=csm,
                    cluster_selection_epsilon=cseps,
                    prediction_data=False,  # avoid warnings when precomputed
                )
                trial_labels = clusterer.fit_predict(dist_matrix)

                n_clusters = len(set(trial_labels)) - (1 if -1 in trial_labels else 0)
                n_noise = int((trial_labels == -1).sum())
                print(
                    f"HDBSCAN mcs={mcs}, ms={ms}, eps={cseps:.3f} -> "
                    f"clusters={n_clusters}, noise={n_noise}"
                )
                s = score_tuple(n_clusters, n_noise)

                if best is None or s < best[0]:
                    best = (s, mcs, ms, cseps, n_clusters, n_noise)
                    best_labels = trial_labels
                    best_clusterer = clusterer

                # Early exit once we hit target exactly at this param level
                if n_clusters == target_clusters:
                    hit_target = True
                    break
            if hit_target:
                break
        if hit_target:
            break

    _, mcs, ms, cseps, k, noise = best
    print(
        f"Selected HDBSCAN -> min_cluster_size={mcs}, min_samples={ms}, "
        f"cluster_selection_epsilon={cseps:.3f} | clusters={k}, noise={noise}"
    )
    labels = best_labels

    # ----------------------------
    # 5) Build catalogue and copy clips
    # ----------------------------
    unique = [l for l in np.unique(labels) if l != -1]
    cluster_to_global = {c: i + 1 for i, c in enumerate(unique)}
    next_gid = len(unique) + 1

    # Assign a global_id to every tracklet (noise becomes its own person id)
    for i, t in enumerate(tracklet_info):
        lab = labels[i]
        if lab == -1:
            t["global_id"] = next_gid
            next_gid += 1
        else:
            t["global_id"] = cluster_to_global[lab]

    # Per-person, which clips & tracklets
    catalogue_dd: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in tracklet_info:
        catalogue_dd[str(t["global_id"])].append(
            {
                "clip_id": t["clip_id"],
                "local_track_id": t["track_id"],
                "frame_ranges": t["frame_ranges"],
                "num_frames": t["num_frames"],
            }
        )

    # Sort appearances for readability
    for gid in catalogue_dd:
        catalogue_dd[gid].sort(
            key=lambda a: (a["clip_id"], a["frame_ranges"][0][0] if a["frame_ranges"] else -1)
        )

    # Copy clips once per person per clip
    for gid, appearances in catalogue_dd.items():
        person_dir = output_root / f"person_{int(gid):03d}"
        person_dir.mkdir(parents=True, exist_ok=True)
        clips = sorted({a["clip_id"] for a in appearances})
        for clip_id in clips:
            src_path = Path(video_paths[clip_id])
            dst_path = person_dir / src_path.name
            try:
                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"[WARN] Failed to copy '{src_path}' to '{dst_path}': {e}")

    # Write compact summary
    summary = {
        "total_unique_persons": len(catalogue_dd),
        "total_tracklets": len(tracklet_info),
        "parameters": {
            "target_clusters": target_clusters,
            "hdbscan_grid": hdbscan_grid,
            "use_median": use_median,
        },
    }
    catalogue = {k: v for k, v in catalogue_dd.items()}
    output = {"summary": summary, "catalogue": catalogue}

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Unique persons: {summary['total_unique_persons']}")
    print(f"Tracklets clustered: {summary['total_tracklets']}")
    print(f"Clips copied into: {output_root.resolve()}")
    print(f"Catalogue saved to: {Path(output_file).resolve()}")

    return catalogue