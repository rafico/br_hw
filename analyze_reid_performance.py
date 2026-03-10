#!/usr/bin/env python3
"""Analyze re-ID performance and suggest improvements."""

import json
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from collections import defaultdict

# Load detections cache
with open('/home/rafi/Downloads/blackrover_hw/videos/.cache/re_id_01b6f3334bf6ba4224064a3f71dbd839_all_detections.json', 'r') as f:
    all_detections = json.load(f)

print(f"Loaded {len(all_detections)} detections")

# Build tracklets
tracklets = defaultdict(list)
for d in all_detections:
    tracklets[(str(d["clip_id"]), int(d["track_id"]))] += [d]

print(f"Found {len(tracklets)} tracklets")

# Compute representative embeddings (using mean, L2-normalized)
tracklet_info = []
for (clip_id, track_id), dets in sorted(tracklets.items()):
    embeds = np.array([det["embeddings"] for det in dets], dtype=np.float32)
    if len(embeds) < 3:  # Skip short tracklets
        continue

    # L2 normalize each, take mean, re-normalize
    embeds_norm = normalize(embeds)
    rep = embeds_norm.mean(axis=0)
    rep = rep / np.linalg.norm(rep)

    tracklet_info.append({
        "clip_id": clip_id,
        "track_id": track_id,
        "embedding": rep,
        "num_frames": len(embeds)
    })

print(f"Kept {len(tracklet_info)} tracklets after filtering")

# Compute pairwise distances
X = np.stack([t["embedding"] for t in tracklet_info], axis=0).astype(np.float32)
X = normalize(X)
dist_matrix = pairwise_distances(X, metric="cosine")

# Analyze distance distribution
print("\n=== Distance Distribution ===")
# Only look at cross-clip pairs
cross_clip_distances = []
same_clip_distances = []

for i in range(len(tracklet_info)):
    for j in range(i + 1, len(tracklet_info)):
        dist = dist_matrix[i, j]
        if tracklet_info[i]["clip_id"] != tracklet_info[j]["clip_id"]:
            cross_clip_distances.append(dist)
        else:
            same_clip_distances.append(dist)

cross_clip_distances = np.array(cross_clip_distances)
same_clip_distances = np.array(same_clip_distances)

print(f"\nCross-clip distances (N={len(cross_clip_distances)}):")
print(f"  Min:     {cross_clip_distances.min():.4f}")
print(f"  10th %:  {np.percentile(cross_clip_distances, 10):.4f}")
print(f"  25th %:  {np.percentile(cross_clip_distances, 25):.4f}")
print(f"  Median:  {np.median(cross_clip_distances):.4f}")
print(f"  75th %:  {np.percentile(cross_clip_distances, 75):.4f}")
print(f"  90th %:  {np.percentile(cross_clip_distances, 90):.4f}")
print(f"  Max:     {cross_clip_distances.max():.4f}")

print(f"\nSame-clip distances (N={len(same_clip_distances)}):")
if len(same_clip_distances) > 0:
    print(f"  Min:     {same_clip_distances.min():.4f}")
    print(f"  Median:  {np.median(same_clip_distances):.4f}")
    print(f"  Max:     {same_clip_distances.max():.4f}")

# Count matches at different thresholds
print("\n=== Potential Matches at Different Thresholds ===")
for threshold in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    n_matches = (cross_clip_distances <= threshold).sum()
    print(f"  Threshold {threshold:.2f}: {n_matches} cross-clip pairs")

# Find closest cross-clip pairs
print("\n=== Top 20 Closest Cross-Clip Pairs ===")
cross_clip_pairs = []
for i in range(len(tracklet_info)):
    for j in range(i + 1, len(tracklet_info)):
        if tracklet_info[i]["clip_id"] != tracklet_info[j]["clip_id"]:
            dist = dist_matrix[i, j]
            cross_clip_pairs.append((dist, i, j))

cross_clip_pairs.sort()
for dist, i, j in cross_clip_pairs[:20]:
    t1, t2 = tracklet_info[i], tracklet_info[j]
    print(f"  Distance {dist:.4f}: clip{t1['clip_id']}_track{t1['track_id']} ({t1['num_frames']}f) <-> "
          f"clip{t2['clip_id']}_track{t2['track_id']} ({t2['num_frames']}f)")

print("\n=== Current vs Recommended Settings ===")
print(f"Current epsilon: 0.2 -> matched {(cross_clip_distances <= 0.2).sum()} pairs")
print(f"Recommended epsilon: 0.35 -> would match {(cross_clip_distances <= 0.35).sum()} pairs")
print(f"Aggressive epsilon: 0.4 -> would match {(cross_clip_distances <= 0.4).sum()} pairs")
