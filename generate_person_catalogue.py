import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from rerank import kreciprocal_rerank


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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


def tracklets_cooccur(t1: dict, t2: dict) -> bool:
    if str(t1["clip_id"]) != str(t2["clip_id"]):
        return False
    for a1, a2 in t1["frame_ranges"]:
        for b1, b2 in t2["frame_ranges"]:
            if a1 <= b2 and b1 <= a2:
                return True
    return False


def _components_cooccur(
        members_a: List[int],
        members_b: List[int],
        tracklet_info: List[Dict[str, Any]],
) -> bool:
    return any(
        tracklets_cooccur(tracklet_info[i], tracklet_info[j])
        for i in members_a
        for j in members_b
    )


def _parse_clip_start_time(clip_id: str) -> Optional[float]:
    """Best-effort parse of absolute start timestamp from clip name."""
    s = str(clip_id)

    # Pattern: YYYYMMDD[_-]HHMMSS or YYYY-MM-DD[_T]HH-MM-SS
    dt_patterns = [
        r"(20\d{2})(\d{2})(\d{2})[_\-T]?(\d{2})(\d{2})(\d{2})",
        r"(20\d{2})[-_](\d{2})[-_](\d{2})[_\-T](\d{2})[-_](\d{2})[-_](\d{2})",
    ]
    for pattern in dt_patterns:
        m = re.search(pattern, s)
        if not m:
            continue
        try:
            parts = [int(g) for g in m.groups()]
            dt = datetime(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
            return dt.timestamp()
        except ValueError:
            continue

    # Pattern: unix timestamp in milliseconds (13 digits) or seconds (10 digits)
    ms_match = re.search(r"(?<!\d)(1\d{12})(?!\d)", s)
    if ms_match:
        return int(ms_match.group(1)) / 1000.0
    sec_match = re.search(r"(?<!\d)(1\d{9})(?!\d)", s)
    if sec_match:
        return float(int(sec_match.group(1)))

    return None


def _smooth_embeddings(embeddings: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Simple moving-average smoothing along time for tracklet embeddings."""
    embeds = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
    n = len(embeds)
    if n < 3 or window_size <= 1:
        return embeds

    w = max(1, int(window_size))
    if w % 2 == 0:
        w += 1
    half = w // 2

    smoothed = np.empty_like(embeds)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed[i] = embeds[lo:hi].mean(axis=0)
    return _normalize_rows(smoothed)


def _build_detection_weights(
        dets: List[Dict[str, Any]],
        quality_alpha: float = 0.75,
) -> np.ndarray:
    """Build representative-aggregation weights using quality/confidence."""
    alpha = float(np.clip(quality_alpha, 0.0, 1.0))
    raw = []
    for det in dets:
        q = _safe_float(det.get("quality"))
        if q is None:
            q = _safe_float(det.get("confidence"))
        if q is None:
            q = 1.0
        raw.append(float(np.clip(q, 0.0, 1.0)))

    quality = np.asarray(raw, dtype=np.float32)
    weights = (1.0 - alpha) + (alpha * quality)
    weights = np.clip(weights, 1e-6, None)
    weights /= np.sum(weights)
    return weights


def _compute_motion_profile(dets: List[Dict[str, Any]]) -> Optional[float]:
    """Estimate motion speed from normalized bbox centers (if available)."""
    frames = []
    centers = []
    for det in dets:
        cx = _safe_float(det.get("center_x"))
        cy = _safe_float(det.get("center_y"))
        if cx is None or cy is None:
            continue
        frames.append(int(det["frame_num"]))
        centers.append((cx, cy))

    if len(frames) < 2:
        return None

    speeds = []
    for idx in range(1, len(frames)):
        dt = max(1, int(frames[idx] - frames[idx - 1]))
        dx = centers[idx][0] - centers[idx - 1][0]
        dy = centers[idx][1] - centers[idx - 1][1]
        speeds.append(float(np.hypot(dx, dy) / dt))

    if not speeds:
        return None
    return float(np.median(np.asarray(speeds, dtype=np.float32)))


def compute_tracklet_representative(
        embeddings: np.ndarray,
        use_median: bool = False,
        weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Representative embedding from tracklet detections.
    - If `weights` is provided, uses weighted mean.
    - Else uses mean or median depending on `use_median`.
    """
    if len(embeddings) == 0:
        raise ValueError("Tracklet has no embeddings")

    embeds = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if len(w) != len(embeds):
            raise ValueError("weights length must match embeddings length")
        w = np.clip(w, 1e-6, None)
        w /= np.sum(w)
        emb = (embeds * w[:, None]).sum(axis=0)
    elif use_median:
        emb = np.median(embeds, axis=0)
    else:
        emb = embeds.mean(axis=0)

    n = np.linalg.norm(emb)
    return emb if n == 0 else emb / n


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


# ----------------------------
# Core pipeline functions
# ----------------------------

def build_tracklets(all_detections: List[Dict[str, Any]]) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    """Group detections by (clip_id, track_id) into tracklets."""
    tracklets = defaultdict(list)
    for d in all_detections:
        tracklets[(str(d["clip_id"]), int(d["track_id"]))] += [d]
    return tracklets


def compute_tracklet_info(
        tracklets: Dict[Tuple[str, int], List[Dict[str, Any]]],
        use_median: bool = False,
        min_tracklet_frames: int = 0,
        use_quality_weights: bool = True,
        quality_alpha: float = 0.75,
        smooth_embeddings: bool = True,
        smoothing_window: int = 5,
) -> List[Dict[str, Any]]:
    """Compute per-tracklet features/metadata for clustering."""
    tracklet_info = []
    dropped = 0

    clip_max_frame: Dict[str, int] = defaultdict(int)
    for (clip_id, _), dets in tracklets.items():
        if dets:
            clip_max_frame[clip_id] = max(
                clip_max_frame[clip_id],
                max(int(det["frame_num"]) for det in dets),
            )

    for (clip_id, track_id), dets in sorted(tracklets.items(), key=lambda x: (x[0][0], x[0][1])):
        dets_sorted = sorted(dets, key=lambda d: int(d["frame_num"]))
        embeds = np.array([det["embeddings"] for det in dets_sorted], dtype=np.float32)
        if len(embeds) < min_tracklet_frames:
            dropped += 1
            continue

        all_embeds = _smooth_embeddings(embeds, smoothing_window) if smooth_embeddings else _normalize_rows(embeds)
        weights = _build_detection_weights(dets_sorted, quality_alpha=quality_alpha) if use_quality_weights else None
        rep = compute_tracklet_representative(
            all_embeds,
            use_median=use_median,
            weights=weights,
        )

        frames = [int(det["frame_num"]) for det in dets_sorted]
        start_frame = int(frames[0])
        end_frame = int(frames[-1])
        center_frame = 0.5 * (start_frame + end_frame)
        denom = max(1, int(clip_max_frame.get(clip_id, end_frame)))
        center_ratio = center_frame / float(denom)

        ts_values = []
        for det in dets_sorted:
            ts = _safe_float(det.get("timestamp_sec"))
            if ts is not None:
                ts_values.append(ts)
        rel_center_sec = float(np.median(ts_values)) if ts_values else None
        clip_start_sec = _parse_clip_start_time(clip_id)
        abs_center_sec = None
        if clip_start_sec is not None and rel_center_sec is not None:
            abs_center_sec = clip_start_sec + rel_center_sec

        quality_values = []
        for det in dets_sorted:
            q = _safe_float(det.get("quality"))
            if q is None:
                q = _safe_float(det.get("confidence"))
            if q is not None:
                quality_values.append(q)
        quality_mean = float(np.mean(quality_values)) if quality_values else None

        tracklet_info.append(
            {
                "clip_id": clip_id,
                "track_id": track_id,
                "embedding": rep,
                "all_embeddings": all_embeds,
                "frame_ranges": _frame_ranges(frames),
                "num_frames": len(frames),
                "temporal_center_ratio": center_ratio,
                "relative_time_center_sec": rel_center_sec,
                "absolute_time_center_sec": abs_center_sec,
                "motion_speed": _compute_motion_profile(dets_sorted),
                "quality_mean": quality_mean,
            }
        )

    if dropped:
        print(f"Dropped {dropped} tracklets with < {min_tracklet_frames} frames")
    return tracklet_info


def build_distance_matrix(
        tracklet_info: List[Dict[str, Any]],
        print_distances: bool = False,
        linkage: str = "min",
        temporal_penalty: float = 0.0,
        temporal_max_gap_sec: Optional[float] = None,
        motion_penalty: float = 0.0,
        use_rerank: bool = False,
) -> np.ndarray:
    """
    Build pairwise distance matrix between tracklets.
    Supports linkage on all embeddings (`min`/`mean`) and representative linkage.
    """
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
    """
    Build candidate pairs via sparse neighbor search.
    For `linkage=min`, this uses all detection embeddings (priority #1).
    """
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

    # For mean/representative linkage, prune candidates with representative search then refine.
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


def _cluster_from_sorted_pairs(
        sorted_pairs: List[Tuple[float, int, int]],
        tracklet_info: List[Dict[str, Any]],
        min_cluster_size: int,
        cluster_selection_epsilon: float,
        cooccurrence_constraint: bool = True,
) -> np.ndarray:
    """Union-find clustering with cross-clip or co-occurrence constraints."""
    n = len(tracklet_info)

    # Union-find structures
    parent = list(range(n))
    rank = [0] * n
    sizes = [1] * n

    # Track the set of clip_ids present in each component (by root index)
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
        # Union by rank
        if rank[px] < rank[py]:
            parent[px] = py
            sizes[py] += sizes[px]
            comp_clip_sets[py] |= comp_clip_sets[px]
            comp_members[py].extend(comp_members[px])
        elif rank[px] > rank[py]:
            parent[py] = px
            sizes[px] += sizes[py]
            comp_clip_sets[px] |= comp_clip_sets[py]
            comp_members[px].extend(comp_members[py])
        else:
            parent[py] = px
            sizes[px] += sizes[py]
            comp_clip_sets[px] |= comp_clip_sets[py]
            comp_members[px].extend(comp_members[py])
            rank[px] += 1
        return True

    # Greedily merge closest pairs if within epsilon and the configured constraint holds.
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
    print("\nClusters:")
    for cluster_id, indices in enumerate(cluster_groups.values()):
        members = [f"{tracklet_info[i]['clip_id']}_{tracklet_info[i]['track_id']}" for i in sorted(indices)]
        print(f"  Cluster {cluster_id}: {{{', '.join(members)}}}")

    # Print noise points with original track_id
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
    """
    Post-process noise labels by merging singleton tracklets into nearest valid cluster.
    Keeps cross-clip constraint intact.
    """
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
    # NOTE: put non-noise after noise handled so `next_gid` stable.
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

    # Sort appearances for readability.
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
    """
      1) Group detections by (clip_id, track_id) => tracklets
      2) Compute weighted/smoothed tracklet representations (priorities #2 and #6)
      3) Build candidate graph (dense or sparse all-embedding linkage)
      4) Cluster with cross-clip constraints (+ temporal/motion penalties)
      5) Post-process singleton merges
      6) Assign IDs and save catalogue
    """
    if not all_detections:
        print("No detections found.")
        return {}

    quality_alpha = float(np.clip(quality_alpha, 0.0, 1.0))
    temporal_penalty = max(0.0, float(temporal_penalty))
    motion_penalty = max(0.0, float(motion_penalty))

    # Build tracklets and compute representatives.
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

    # Assign person IDs and build catalogue.
    assign_person_ids(tracklet_info, labels)
    catalogue = build_catalogue(tracklet_info)

    # Save catalogue.
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

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Unique persons: {summary['total_unique_persons']}")
    print(f"Tracklets clustered: {summary['total_tracklets']}")
    print(f"Catalogue saved to: {Path(output_file).resolve()}")

    return catalogue
