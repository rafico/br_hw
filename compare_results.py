#!/usr/bin/env python3
"""Compare old vs new re-ID results."""

import json
import sys
from collections import Counter
from pathlib import Path


def analyze_catalogue(filepath, label):
    """Analyze and print catalogue statistics."""
    with Path(filepath).open("r", encoding="utf-8") as f:
        data = json.load(f)

    catalogue = data["catalogue"]
    params = data["summary"]

    print(f"\n{'='*60}")
    print(label)
    print(f"{'='*60}")

    print("\nParameters:")
    for k, v in params["parameters"].items():
        print(f"  {k}: {v}")

    print("\nSummary:")
    print(f"  Total unique persons: {params['total_unique_persons']}")
    print(f"  Total tracklets: {params['total_tracklets']}")

    # Analyze cross-clip matches
    cross_clip = []
    single_clip = []
    total_appearances = 0

    for person_id, appearances in catalogue.items():
        clips = {app["clip_id"] for app in appearances}
        total_appearances += len(appearances)
        if len(clips) > 1:
            cross_clip.append((person_id, len(clips), clips, appearances))
        else:
            single_clip.append(person_id)

    avg_appearances = total_appearances / len(catalogue) if catalogue else 0.0

    print("\nCross-clip analysis:")
    print(f"  Persons appearing in multiple clips: {len(cross_clip)}")
    print(f"  Single-clip persons: {len(single_clip)}")
    print(f"  Average appearances per person: {avg_appearances:.2f}")

    if cross_clip:
        print("\n  Cross-clip matches (detailed):")
        for pid, n_clips, clips, apps in sorted(cross_clip, key=lambda x: -x[1]):
            clips_str = ", ".join(sorted(clips))
            total_frames = sum(app["num_frames"] for app in apps)
            print(f"    Person {pid}: {n_clips} clips [{clips_str}] ({total_frames} total frames)")

    # Tracklets per clip
    clip_counts = Counter()
    for appearances in catalogue.values():
        for app in appearances:
            clip_counts[app["clip_id"]] += 1

    print("\n  Tracklets per clip:")
    for clip in sorted(clip_counts.keys()):
        print(f"    Clip {clip}: {clip_counts[clip]} tracklets")

    return {
        "total_persons": params["total_unique_persons"],
        "cross_clip_persons": len(cross_clip),
        "single_clip_persons": len(single_clip),
        "total_tracklets": params["total_tracklets"],
    }


if __name__ == "__main__":
    # Check if new catalogue exists
    old_file = Path("catalogue_simple.json")

    if not old_file.exists():
        print("Error: catalogue_simple.json not found")
        sys.exit(1)

    # For now, just analyze current
    stats = analyze_catalogue(old_file, "CURRENT RESULTS")

    print(f"\n{'='*60}")
    print("IMPROVEMENT METRICS")
    print(f"{'='*60}")
    print(f"\nCross-clip match rate: {stats['cross_clip_persons']} / {stats['total_persons']} "
          f"= {100 * stats['cross_clip_persons'] / stats['total_persons']:.1f}%")

    print("\nExpected improvements:")
    print("  OLD (baseline): ~2 cross-clip persons out of 25 (8%)")
    print("  TARGET: 8-12 cross-clip persons out of 15-18 total (50-60%)")
