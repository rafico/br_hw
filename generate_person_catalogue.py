import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import shutil
from sklearn.cluster import DBSCAN
import numpy as np
import hdbscan
from sklearn.preprocessing import normalize

def _normalize_rows(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)

def compute_tracklet_representative(embeddings):
    """
    Simple representative embedding: L2-normalize each, take mean, re-normalize.
    """
    if len(embeddings) == 0:
        raise ValueError("Tracklet has no embeddings")
    emb = _normalize_rows(np.asarray(embeddings)).mean(axis=0)
    n = np.linalg.norm(emb)
    return emb if n == 0 else emb / n

def _frame_ranges(frame_list):
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

def _build_video_paths_from_dataset(dataset):
    """
    Build {clip_id: filepath} from a FiftyOne dataset (clip_id = stem of filepath).
    """
    video_paths = {}
    for sample in dataset:
        clip_id = Path(sample.filepath).stem
        video_paths[clip_id] = sample.filepath
    return video_paths

def generate_person_catalogue_and_save_clips(
    all_detections,
    dataset=None,
    video_paths=None,
    output_dir="persons",
    output_file="catalogue_simple.json",
    eps=0.35,
    min_samples=2,
):
    """
    Minimal pipeline:
      1) Group detections by (clip_id, track_id) => tracklets
      2) Compute one representative embedding per tracklet (no outlier removal)
      3) Cluster tracklets across clips (DBSCAN w/ cosine)
      4) For each person, copy *entire* clip files they appear in into output_dir/person_XXX/

    Params:
      - all_detections: list of dicts with keys: clip_id, track_id, frame_num, embeddings
      - dataset: optional FiftyOne dataset, used to resolve clip filepaths
      - video_paths: optional dict {clip_id: absolute_filepath}; used if dataset is None
      - output_dir: root directory to place person folders
      - output_file: JSON path with a compact catalogue
      - eps, min_samples: DBSCAN params (metric='cosine')
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
        tracklets[(d["clip_id"], d["track_id"])].append(d)
    print(f"Found {len(tracklets)} tracklets")

    # 2) Representatives
    tracklet_info = []
    for (clip_id, track_id), dets in tracklets.items():
        embeds = np.array([det["embeddings"] for det in dets])
        rep = compute_tracklet_representative(embeds)
        frames = sorted([det["frame_num"] for det in dets])
        tracklet_info.append({
            "clip_id": clip_id,
            "track_id": track_id,
            "embedding": rep,
            "frame_ranges": _frame_ranges(frames),
            "num_frames": len(frames),
        })

    if not tracklet_info:
        print("No tracklet info produced.")
        return {}

    # 3) Cluster tracklets (same person across clips) — HDBSCAN
    X = np.stack([t["embedding"] for t in tracklet_info], axis=0).astype(np.float32)
    X = normalize(X)  # important for cosine distance
    print(X.shape)

    # Sweep a few gentle settings and report results (similar to your eps sweep)
    target_clusters = 7
    best = None
    best_labels = None
    best_clusterer = None

    for min_cluster_size in [3, 4, 5, 6]:
        for min_samples in [2, 3, 4]:
            for cseps in [0.0, 0.005, 0.01, 0.015, 0.02]:
                clusterer = hdbscan.HDBSCAN(
                    metric="cosine",
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method="leaf",  # finer clusters
                    cluster_selection_epsilon=cseps,
                    prediction_data=True
                )
                labels = clusterer.fit_predict(X)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = int((labels == -1).sum())
                print(f"mcs={min_cluster_size}, ms={min_samples}, eps={cseps:.3f} -> "
                      f"clusters={n_clusters}, noise={n_noise}")

                # pick params that aim for ~7 clusters while keeping noise modest
                score = abs(n_clusters - target_clusters) + 0.25 * (n_noise / len(X))
                if best is None or score < best[0]:
                    best = (score, min_cluster_size, min_samples, cseps, n_clusters, n_noise)
                    best_labels = labels
                    best_clusterer = clusterer

    # Final selection
    _, mcs, ms, cseps, k, noise = best
    print(f"Selected HDBSCAN -> min_cluster_size={mcs}, min_samples={ms}, "
          f"cluster_selection_epsilon={cseps:.3f} | clusters={k}, noise={noise}")

    labels = best_labels
    clusterer = best_clusterer

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clusterer.fit_predict(X)

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

    # 4) Build simple catalogue: per person, which clips & tracklets
    catalogue = defaultdict(list)
    for t in tracklet_info:
        catalogue[str(t["global_id"])].append({
            "clip_id": t["clip_id"],
            "local_track_id": t["track_id"],
            "frame_ranges": t["frame_ranges"],
            "num_frames": t["num_frames"],
        })

    # Sort appearances for readability
    for gid in catalogue:
        catalogue[gid].sort(key=lambda a: (a["clip_id"], a["frame_ranges"][0][0] if a["frame_ranges"] else -1))

    # Copy each relevant clip into that person's folder
    for gid, appearances in catalogue.items():
        person_dir = output_root / f"person_{int(gid):03d}"
        person_dir.mkdir(parents=True, exist_ok=True)

        # Unique clips for this person
        clip_ids = sorted(set(a["clip_id"] for a in appearances))
        for cid in clip_ids:
            src = video_paths.get(cid)
            if not src:
                print(f"[WARN] Missing video path for clip_id '{cid}'")
                continue
            src_path = Path(src)
            dst_path = person_dir / src_path.name
            if not dst_path.exists():
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"[WARN] Failed to copy '{src_path}' -> '{dst_path}': {e}")

    # Write compact summary
    summary = {
        "total_unique_persons": len(catalogue),
        "total_tracklets": len(tracklet_info),
        "parameters": {"dbscan_eps": eps, "dbscan_min_samples": min_samples},
    }
    output = {"summary": summary, "catalogue": catalogue}

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Unique persons: {summary['total_unique_persons']}")
    print(f"Tracklets clustered: {summary['total_tracklets']}")
    print(f"Clips copied into: {output_root.resolve()}")
    print(f"Catalogue saved to: {Path(output_file).resolve()}")

    return catalogue
