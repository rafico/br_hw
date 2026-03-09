import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def compute_tracklet_representative(embeddings: np.ndarray, use_median: bool = False) -> np.ndarray:
    """
    Simple representative embedding: L2-normalize each, take mean or median, re-normalize.
    Returns a 1D float32 vector.
    """
    if len(embeddings) == 0:
        raise ValueError("Tracklet has no embeddings")
    embeds = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
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


def _check_distance_matrix(D: np.ndarray):
    """Basic sanity checks to catch issues early."""
    if not np.isfinite(D).all():
        raise ValueError("Distance matrix contains non-finite values.")
    if (D < 0).any():
        raise ValueError("Distance matrix has negative entries.")
    if not np.allclose(D, D.T, atol=1e-6):
        raise ValueError("Distance matrix must be symmetric.")


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
                          use_median: bool = False,
                          min_tracklet_frames: int = 0) -> List[Dict[str, Any]]:
    """Compute representative embeddings and frame ranges for each tracklet.

    Also stores all per-detection embeddings for min-linkage distance computation.
    Tracklets with fewer than min_tracklet_frames are dropped.
    """
    tracklet_info = []
    dropped = 0
    for (clip_id, track_id), dets in sorted(tracklets.items(), key=lambda x: (x[0][0], x[0][1])):
        embeds = np.array([det["embeddings"] for det in dets], dtype=np.float32)
        if len(embeds) < min_tracklet_frames:
            dropped += 1
            continue
        rep = compute_tracklet_representative(embeds, use_median=use_median)
        all_embeds = _normalize_rows(embeds)
        frames = sorted([int(det["frame_num"]) for det in dets])
        tracklet_info.append(
            {
                "clip_id": clip_id,
                "track_id": track_id,
                "embedding": rep,
                "all_embeddings": all_embeds,
                "frame_ranges": _frame_ranges(frames),
                "num_frames": len(frames),
            }
        )
    if dropped:
        print(f"Dropped {dropped} tracklets with < {min_tracklet_frames} frames")
    return tracklet_info


def build_distance_matrix(tracklet_info: List[Dict[str, Any]],
                          print_distances: bool = False,
                          linkage: str = "min",
                          ) -> np.ndarray:
    """
    Build cosine distance matrix between tracklets.

    linkage controls how multi-detection tracklets are compared:
      - "min": minimum pairwise distance (single-linkage, best for re-ID)
      - "mean": mean of all pairwise distances
      - "representative": original approach using single representative embedding

    Prints tracklet order and pairwise distances for debugging.
    """
    n = len(tracklet_info)
    if print_distances:
        print("Tracklet order:")
        for i, t in enumerate(tracklet_info):
            print(f"  {i}: {t['clip_id']}_{t['track_id']} ({t['num_frames']} frames)")

    if linkage == "representative":
        X = np.stack([t["embedding"] for t in tracklet_info], axis=0).astype(np.float32)
        X = normalize(X)
        dist_matrix = pairwise_distances(X, metric="cosine")
    else:
        dist_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                # Compute all pairwise cosine distances between detections
                D = pairwise_distances(
                    tracklet_info[i]["all_embeddings"],
                    tracklet_info[j]["all_embeddings"],
                    metric="cosine",
                )
                if linkage == "min":
                    d = float(D.min())
                else:  # mean
                    d = float(D.mean())
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.clip(dist_matrix, 0.0, 2.0)
    dist_matrix = np.ascontiguousarray(dist_matrix, dtype=np.float32)

    if print_distances:
        print("Pairwise cosine distances between tracklets:")
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
) -> Tuple[List[Tuple[float, int, int]], int, float]:
    """
    Build candidate pairs via radius neighbors to avoid dense O(n^2) pair materialization.
    Uses representative embeddings for the initial radius search, then refines
    with min-linkage distance for candidate pairs when linkage != "representative".
    Returns sorted (dist, i, j) pairs, number of possible pairs, and threshold used.
    """
    n = len(tracklet_info)
    if n < 2:
        return [], 0, cluster_selection_epsilon * 1.1

    X = np.stack([t["embedding"] for t in tracklet_info], axis=0).astype(np.float32)
    X = normalize(X)
    # Use a wider radius for initial candidate search when using min-linkage,
    # since min-linkage distances are typically smaller than representative distances
    search_radius = cluster_selection_epsilon * 1.5 if linkage != "representative" else cluster_selection_epsilon * 1.1
    threshold = cluster_selection_epsilon * 1.1
    total_possible = (n * (n - 1)) // 2

    nn = NearestNeighbors(metric="cosine", algorithm="brute", radius=search_radius)
    nn.fit(X)
    distances, neighbors = nn.radius_neighbors(X, return_distance=True)

    pairs: List[Tuple[float, int, int]] = []
    for src, (dists, nbrs) in enumerate(zip(distances, neighbors)):
        for dist, dst in zip(dists, nbrs):
            dst_i = int(dst)
            if dst_i <= src:
                continue
            if linkage != "representative":
                D = pairwise_distances(
                    tracklet_info[src]["all_embeddings"],
                    tracklet_info[dst_i]["all_embeddings"],
                    metric="cosine",
                )
                if linkage == "min":
                    dist = float(D.min())
                else:
                    dist = float(D.mean())
            if dist <= threshold:
                pairs.append((dist, int(src), dst_i))

    pairs.sort()
    return pairs, total_possible, threshold


def _cluster_from_sorted_pairs(
        sorted_pairs: List[Tuple[float, int, int]],
        tracklet_info: List[Dict[str, Any]],
        min_cluster_size: int,
        cluster_selection_epsilon: float,
) -> np.ndarray:
    """Union-find clustering with cross-clip constraint using pre-sorted candidate pairs."""
    n = len(tracklet_info)

    # Union-find structures
    parent = list(range(n))
    rank = [0] * n
    sizes = [1] * n

    # Track the set of clip_ids present in each component (by root index)
    clip_ids = [str(t["clip_id"]) for t in tracklet_info]
    comp_clip_sets: List[set] = [set([clip_ids[idx]]) for idx in range(n)]

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px = find(x)
        py = find(y)
        if px == py:
            return False
        # Union by rank
        if rank[px] < rank[py]:
            parent[px] = py
            sizes[py] += sizes[px]
            comp_clip_sets[py] |= comp_clip_sets[px]
        elif rank[px] > rank[py]:
            parent[py] = px
            sizes[px] += sizes[py]
            comp_clip_sets[px] |= comp_clip_sets[py]
        else:
            parent[py] = px
            sizes[px] += sizes[py]
            comp_clip_sets[px] |= comp_clip_sets[py]
            rank[px] += 1
        return True

    # Greedily merge closest pairs if within epsilon, and CROSS-CLIP constraint holds
    for dist, a, b in sorted_pairs:
        if dist > cluster_selection_epsilon:
            break
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if not comp_clip_sets[ra].isdisjoint(comp_clip_sets[rb]):
            continue
        union(ra, rb)

    # Collect components
    components = defaultdict(list)
    for idx in range(n):
        root = find(idx)
        components[root].append(idx)

    # Filter clusters by min_cluster_size
    cluster_groups = {k: v for k, v in components.items() if len(v) >= min_cluster_size}

    # Assign labels (-1 for noise/singletons)
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

    # Print clusters with original track_id
    print("\nClusters (cross-clip):")
    for cluster_id, indices in enumerate(cluster_groups.values()):
        members = [f"{tracklet_info[i]['clip_id']}_{tracklet_info[i]['track_id']}" for i in sorted(indices)]
        print(f"  Cluster {cluster_id}: {{{', '.join(members)}}}")

    # Print noise points with original track_id
    noise_indices = np.where(labels == -1)[0]
    if len(noise_indices) > 0:
        noise_members = [f"{tracklet_info[i]['clip_id']}_{tracklet_info[i]['track_id']}" for i in sorted(noise_indices)]
        print(f"  Noise: {{{', '.join(noise_members)}}}")

    return labels


def cluster_tracklets(dist_matrix: np.ndarray,
                      tracklet_info: List[Dict[str, Any]],
                      min_cluster_size: int = 2,
                      cluster_selection_epsilon: float = 0.025,
                      ) -> np.ndarray:
    """
    Cluster tracklets using a custom greedy algorithm.
    Enforces CROSS-CLIP constraint — a cluster may contain at most one
    tracklet from any given clip_id. This is stronger than pairwise cannot-links
    because it prevents transitive merges from introducing duplicate clip_ids.
    Prints clustering details for debugging.
    Returns an array of cluster labels (-1 for noise/singletons).
    """
    print(
        f"Using custom greedy clustering, epsilon={cluster_selection_epsilon} (cross-clip only)"
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
    )


def cluster_tracklets_sparse(
        tracklet_info: List[Dict[str, Any]],
        min_cluster_size: int = 2,
        cluster_selection_epsilon: float = 0.025,
        linkage: str = "min",
) -> np.ndarray:
    """Sparse radius-neighbor variant for large tracklet counts."""
    print(
        f"Using sparse radius-neighbor clustering, epsilon={cluster_selection_epsilon}, "
        f"linkage={linkage} (cross-clip only)"
    )
    sorted_pairs, total_pairs, threshold = build_sparse_candidate_pairs(
        tracklet_info, cluster_selection_epsilon, linkage=linkage
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
    )


def assign_person_ids(tracklet_info: List[Dict[str, Any]], labels: np.ndarray):
    """Assign global person IDs to tracklets based on cluster labels. Modifies tracklet_info in place."""
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

def generate_person_catalogue(
        all_detections: List[Dict[str, Any]],
        output_file: str = "catalogue_simple.json",
        min_cluster_size: int = 2,
        cluster_selection_epsilon: float = 0.2,
        use_median: bool = True,
        print_distances: bool = False,
        use_sparse_neighbors: bool = True,
        sparse_if_n_ge: int = 1000,
        linkage: str = "min",
        min_tracklet_frames: int = 3,
):
    """
      1) Group detections by (clip_id, track_id) => tracklets
      2) Compute one representative embedding per tracklet
      3) Build candidate graph (dense or sparse radius-neighbor)
      4) Cluster tracklets with custom greedy algorithm that
         enforces cross-clip constraint during merges
      5) Assign IDs and build catalogue (and save to JSON)
    Prints progress and summary information.
    Returns the catalogue.
    """
    if not all_detections:
        print("No detections found.")
        return {}

    # Build tracklets and compute representatives
    tracklets = build_tracklets(all_detections)
    print(f"Found {len(tracklets)} tracklets")

    tracklet_info = compute_tracklet_info(tracklets, use_median=use_median,
                                          min_tracklet_frames=min_tracklet_frames)
    if not tracklet_info:
        print("No tracklet info produced.")
        return {}

    n_tracklets = len(tracklet_info)
    use_sparse_path = use_sparse_neighbors and n_tracklets >= sparse_if_n_ge and not print_distances
    if use_sparse_path:
        labels = cluster_tracklets_sparse(
            tracklet_info,
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
            linkage=linkage,
        )
    else:
        dist_matrix = build_distance_matrix(tracklet_info, print_distances=print_distances,
                                            linkage=linkage)
        if print_distances:
            print(f"dist_matrix: {dist_matrix}")
        labels = cluster_tracklets(
            dist_matrix,
            tracklet_info,
            min_cluster_size=min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )

    # Assign person IDs and build catalogue
    assign_person_ids(tracklet_info, labels)
    catalogue = build_catalogue(tracklet_info)

    # Save catalogue
    summary = {
        "total_unique_persons": len(catalogue),
        "total_tracklets": len(tracklet_info),
        "parameters": {
            "min_cluster_size": min_cluster_size,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "use_median": use_median,
            "cross_clip_only": True,
            "linkage": linkage,
            "min_tracklet_frames": min_tracklet_frames,
            "sparse_neighbors_enabled": use_sparse_neighbors,
            "sparse_if_n_ge": sparse_if_n_ge,
            "sparse_path_used": use_sparse_path,
        },
    }
    output = {"summary": summary, "catalogue": catalogue}

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Unique persons: {summary['total_unique_persons']}")
    print(f"Tracklets clustered: {summary['total_tracklets']}")
    print(f"Catalogue saved to: {Path(output_file).resolve()}")

    return catalogue
