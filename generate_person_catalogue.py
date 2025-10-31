import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import hdbscan


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
    """Convert sorted list of frame numbers into [start, end] ranges."""
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
    """Build {clip_id: filepath} from a FiftyOne dataset."""
    video_paths = {}
    for sample in dataset:
        clip_id = Path(sample.filepath).stem
        video_paths[clip_id] = sample.filepath
    return video_paths


def _check_distance_matrix(D: np.ndarray):
    """Basic sanity checks to catch issues early."""
    if not np.isfinite(D).all():
        raise ValueError("Distance matrix contains non-finite values.")
    if (D < 0).any():
        raise ValueError("Distance matrix has negative entries.")
    if not np.allclose(D, D.T, atol=1e-8):
        raise ValueError("Distance matrix must be symmetric.")


# ----------------------------
# NEW: cannot-link from co-occurrence
# ----------------------------

def build_cannot_links_from_detections(
    all_detections: List[Dict[str, Any]],
    tracklet_info: List[Dict[str, Any]]
) -> List[Tuple[int, int]]:
    """
    Returns list of index pairs (i, j) such that tracklets i and j
    have detections in the SAME (clip_id, frame_num) -> cannot be the same person.
    Only pairs that survived into `tracklet_info` are returned.
    """
    # map (clip_id, track_id) -> index in tracklet_info
    id2idx: Dict[Tuple[str, int], int] = {
        (t["clip_id"], t["track_id"]): i for i, t in enumerate(tracklet_info)
    }

    # map (clip_id, frame_num) -> set of indices present in that frame
    cooc: Dict[Tuple[str, int], set] = defaultdict(set)
    for d in all_detections:
        clip = str(d["clip_id"])
        tid = int(d["track_id"])
        frm = int(d["frame_num"])
        key = (clip, tid)
        if key not in id2idx:
            continue  # tracklet filtered out upstream
        cooc[(clip, frm)].add(id2idx[key])

    pairs = set()
    for _, idxs in cooc.items():
        if len(idxs) > 1:
            s = sorted(idxs)
            for a in range(len(s)):
                for b in range(a + 1, len(s)):
                    pairs.add((s[a], s[b]))
    return sorted(pairs)


def apply_cannot_links_to_distance_matrix(
    D: np.ndarray,
    cannot_pairs: List[Tuple[int, int]],
    max_distance: float = 2.0
) -> None:
    """
    In-place: push distances for cannot-link pairs to the maximum cosine distance.
    """
    for i, j in cannot_pairs:
        D[i, j] = max_distance
        D[j, i] = max_distance


# ----------------------------
# Core pipeline functions
# ----------------------------

def build_tracklets(all_detections: List[Dict[str, Any]]) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    """Group detections by (clip_id, track_id) into tracklets."""
    tracklets = defaultdict(list)
    for d in all_detections:
        tracklets[(str(d["clip_id"]), int(d["track_id"]))] += [d]
    return tracklets


def compute_tracklet_info(tracklets: Dict[Tuple[str, int], List[Dict[str, Any]]],
                          use_median: bool = False) -> List[Dict[str, Any]]:
    """Compute representative embeddings and frame ranges for each tracklet."""
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
    return tracklet_info


def build_distance_matrix(tracklet_info: List[Dict[str, Any]],
                          print_distances: bool = False
                          ) -> np.ndarray:
    """Build cosine distance matrix from tracklet embeddings."""
    print("Tracklet order:")
    for i, t in enumerate(tracklet_info):
        print(f"  {i}: {t['clip_id']}_{t['track_id']}")

    X = np.stack([t["embedding"] for t in tracklet_info], axis=0).astype(np.float64)
    X = normalize(X)

    dist_matrix = pairwise_distances(X, metric="cosine")
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, 2.0)
    dist_matrix = np.ascontiguousarray(dist_matrix, dtype=np.float64)

    print("Pairwise cosine distances between tracklet representatives:")
    print(dist_matrix)

    _check_distance_matrix(dist_matrix)
    return dist_matrix


def cluster_tracklets(dist_matrix: np.ndarray,
                      tracklet_info: List[Dict[str, Any]],
                      min_cluster_size: int = 2,
                      min_samples: int = 1,
                      cluster_selection_epsilon: float = 0.025,  # VERY CONSERVATIVE
                      cluster_selection_method: str = "eom") -> np.ndarray:
    """
    Cluster tracklets using HDBSCAN with conservative settings.
    """
    print(f"Using cluster_selection_method: {cluster_selection_method}")
    print(f"Using cluster_selection_epsilon: {cluster_selection_epsilon}")

    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=False,
        alpha=1.0,
    )

    labels = clusterer.fit_predict(dist_matrix)

    print(f"Raw labels: {list(labels)}")

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    total_persons = n_clusters + n_noise

    print(
        f"HDBSCAN mcs={min_cluster_size}, ms={min_samples}, eps={cluster_selection_epsilon:.3f} -> "
        f"clusters={n_clusters}, noise={n_noise}, total_persons={total_persons}"
    )

    # Print clusters with original track_id
    cluster_groups = defaultdict(list)
    for i, label in enumerate(labels):
        if label != -1:
            cluster_groups[label].append(i)

    print("\nClusters:")
    for cluster_id in sorted(cluster_groups.keys()):
        indices = cluster_groups[cluster_id]
        members = [f"{tracklet_info[i]['clip_id']}_{tracklet_info[i]['track_id']}" for i in indices]
        print(f"  Cluster {cluster_id}: {{{', '.join(members)}}}")

    # Print noise points with original track_id
    noise_indices = [i for i, label in enumerate(labels) if label == -1]
    if noise_indices:
        noise_members = [f"{tracklet_info[i]['clip_id']}_{tracklet_info[i]['track_id']}" for i in noise_indices]
        print(f"  Noise: {{{', '.join(noise_members)}}}")

    return labels


def assign_person_ids(tracklet_info: List[Dict[str, Any]], labels: np.ndarray):
    """Assign global person IDs to tracklets based on cluster labels."""
    unique = [l for l in np.unique(labels) if l != -1]
    cluster_to_global = {c: i + 1 for i, c in enumerate(unique)}
    next_gid = len(unique) + 1

    for i, t in enumerate(tracklet_info):
        lab = labels[i]
        if lab == -1:
            t["global_id"] = next_gid
            next_gid += 1
    # NOTE: put non-noise after noise handled so `next_gid` stable
    for i, t in enumerate(tracklet_info):
        lab = labels[i]
        if lab != -1:
            t["global_id"] = cluster_to_global[lab]


def build_catalogue(tracklet_info: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Build per-person catalogue from tracklet info."""
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

    return dict(catalogue_dd)


# ----------------------------
# Main pipeline
# ----------------------------

def generate_person_catalogue_and_save_clips(
        all_detections: List[Dict[str, Any]],
        dataset: Optional[Any] = None,
        video_paths: Optional[Dict[str, str]] = None,
        output_dir: str = "persons",
        output_file: str = "catalogue_simple.json",
        min_cluster_size: int = 2,
        min_samples: int = 1,
        cluster_selection_epsilon: float = 0.025,
        cluster_selection_method: str = "eom",
        use_median: bool = False,
):
    """
      1) Group detections by (clip_id, track_id) => tracklets
      2) Compute one representative embedding per tracklet
      3) Build cosine distance matrix
      4) set distances of co-occurring tracklet pairs to max (2.0)
      5) Cluster tracklets with HDBSCAN
      6) Assign IDs and build/save catalogue
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

    # Build tracklets and compute representatives
    tracklets = build_tracklets(all_detections)
    print(f"Found {len(tracklets)} tracklets")

    tracklet_info = compute_tracklet_info(tracklets, use_median=use_median)
    if not tracklet_info:
        print("No tracklet info produced.")
        return {}

    # Build distance matrix
    dist_matrix = build_distance_matrix(tracklet_info)

    # --- NEW: apply cannot-links from co-occurrence ---
    cannot_pairs = build_cannot_links_from_detections(all_detections, tracklet_info)
    if cannot_pairs:
        print(f"Cannot-links from co-occurrence: {len(cannot_pairs)} pairs")
        apply_cannot_links_to_distance_matrix(dist_matrix, cannot_pairs, max_distance=2.0)
    # ---------------------------------------------------

    print(f'dist_matrix after: {dist_matrix}')

    # Cluster
    labels = cluster_tracklets(
        dist_matrix,
        tracklet_info,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method
    )

    # Assign person IDs and build catalogue
    assign_person_ids(tracklet_info, labels)
    catalogue = build_catalogue(tracklet_info)

    # Save catalogue (same filenames)
    summary = {
        "total_unique_persons": len(catalogue),
        "total_tracklets": len(tracklet_info),
        "parameters": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "cluster_selection_method": cluster_selection_method,
            "use_median": use_median,
        },
    }
    output = {"summary": summary, "catalogue": catalogue}

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Unique persons: {summary['total_unique_persons']}")
    print(f"Tracklets clustered: {summary['total_tracklets']}")
    print(f"Clips copied into: {output_root.resolve()}")
    print(f"Catalogue saved to: {Path(output_file).resolve()}")

    return catalogue