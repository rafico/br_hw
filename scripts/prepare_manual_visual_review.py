from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_repo_root_on_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Prepare artifacts and notes for the manual visual QA checklist.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory used by the pipeline run")
    parser.add_argument("--detections-cache", default="", help="Optional explicit detections cache path")
    parser.add_argument("--catalogue", default="catalogue_v2.json", help="Catalogue JSON to review")
    parser.add_argument("--scene", default="scene_labels_v2.json", help="Scene labels JSON to review")
    parser.add_argument("--rerun-recording", default="qa_artifacts/recording.rrd", help="Rerun recording to review")
    parser.add_argument("--manifest-out", default="qa_artifacts/manual_visual_review.json", help="Output JSON manifest path")
    parser.add_argument("--notes-out", default="qa_artifacts/manual_visual_review.md", help="Output markdown notes path")
    parser.add_argument("--sample-frames", type=int, default=10, help="Number of review frames to sample")
    parser.add_argument("--consistency-frames", type=int, default=5, help="Number of review frames to earmark for cross-artifact checks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_root_on_path()
    from qa.manual_review import build_review_manifest, load_review_inputs, render_review_notes

    args = parse_args(argv)
    detections, catalogue_payload, scene_payload, detections_cache_path = load_review_inputs(
        dataset_dir=args.dataset_dir,
        detections_cache=args.detections_cache,
        catalogue=args.catalogue,
        scene=args.scene,
        rerun_recording=args.rerun_recording,
    )

    manifest = build_review_manifest(
        dataset_dir=args.dataset_dir,
        detections=detections,
        catalogue_payload=catalogue_payload,
        scene_payload=scene_payload,
        detections_cache_path=str(detections_cache_path),
        catalogue_path=args.catalogue,
        scene_path=args.scene,
        rerun_recording_path=args.rerun_recording,
        sample_count=args.sample_frames,
        consistency_count=args.consistency_frames,
    )

    manifest_path = Path(args.manifest_out).expanduser().resolve()
    notes_path = Path(args.notes_out).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    notes_path.write_text(render_review_notes(manifest), encoding="utf-8")

    print(f"[manual-review] manifest: {manifest_path}")
    print(f"[manual-review] notes: {notes_path}")
    print(f"[manual-review] sampled frames: {manifest['summary']['review_frame_count']}")
    print(f"[manual-review] clips covered: {manifest['summary']['clip_count']}")
    if manifest["warnings"]:
        print(f"[manual-review] warnings: {len(manifest['warnings'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
