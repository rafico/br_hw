from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np


def assign_person_ids(tracklet_info: List[dict], labels: np.ndarray):
    """Assign global person IDs to tracklets based on cluster labels."""
    unique = [l for l in np.unique(labels) if l != -1]
    cluster_to_global = {c: i + 1 for i, c in enumerate(unique)}
    next_gid = len(unique) + 1

    for i, t in enumerate(tracklet_info):
        lab = labels[i]
        if lab == -1:
            t["global_id"] = next_gid
            next_gid += 1
    for i, t in enumerate(tracklet_info):
        lab = labels[i]
        if lab != -1:
            t["global_id"] = cluster_to_global[lab]


def build_catalogue(tracklet_info: List[dict]) -> Dict[str, List[dict]]:
    """Build per-person catalogue from tracklet info."""
    catalogue_dd: Dict[str, List[dict]] = defaultdict(list)
    for t in tracklet_info:
        catalogue_dd[str(t["global_id"])].append(
            {
                "clip_id": t["clip_id"],
                "local_track_id": t["track_id"],
                "frame_ranges": t["frame_ranges"],
                "num_frames": t["num_frames"],
            }
        )

    for gid in catalogue_dd:
        catalogue_dd[gid].sort(
            key=lambda a: (a["clip_id"], a["frame_ranges"][0][0] if a["frame_ranges"] else -1)
        )

    return dict(catalogue_dd)
