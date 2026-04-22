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


_ensure_repo_root_on_path()

from qa.output_validation import validate_catalogue_payload, validate_scene_payload


def _is_json_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _raise_if_invalid(label: str, errors: list[str]) -> None:
    if errors:
        raise ValueError(f"{label} payload failed validation:\n- " + "\n- ".join(errors))


def build_ground_truth(catalogue_payload: dict, scene_payload: list[dict]) -> dict:
    _raise_if_invalid("catalogue", validate_catalogue_payload(catalogue_payload))
    _raise_if_invalid("scene", validate_scene_payload(scene_payload))

    persons = []
    for global_id in sorted(catalogue_payload.get("catalogue", {}), key=lambda value: int(value)):
        appearances = []
        for appearance in catalogue_payload["catalogue"][global_id]:
            appearances.append(
                {
                    "clip": str(appearance["clip_id"]),
                    "frame_ranges": appearance.get("frame_ranges", []),
                }
            )
        persons.append(
            {
                "global_id": int(global_id),
                "appearances": appearances,
            }
        )

    scenes = []
    for scene in sorted(scene_payload, key=lambda item: str(item.get("clip_id", ""))):
        people = sorted(
            {
                int(person_id)
                for segment in scene.get("crime_segments", [])
                for person_id in segment.get("involved_people_global", [])
                if _is_json_int(person_id)
            }
        )
        scenes.append(
            {
                "clip": str(scene.get("clip_id", "")),
                "label": str(scene.get("label", "normal")),
                "crime_spans_sec": [
                    [float(segment.get("timestamp_start", 0.0)), float(segment.get("timestamp_end", 0.0))]
                    for segment in scene.get("crime_segments", [])
                ],
                "crime_person_global_ids": people,
            }
        )

    return {
        "_template_note": (
            "Seeded from predicted catalogue and scene labels. "
            "Review and correct every person ID, span, and scene label by hand before evaluation."
        ),
        "persons": persons,
        "scenes": scenes,
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Seed ground_truth.json from current predictions")
    parser.add_argument("--catalogue", default="catalogue_v2.json")
    parser.add_argument("--scenes", default="scene_labels_v2.json")
    parser.add_argument("--output", default="ground_truth.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    with open(args.catalogue, "r", encoding="utf-8") as f:
        catalogue_payload = json.load(f)
    with open(args.scenes, "r", encoding="utf-8") as f:
        scene_payload = json.load(f)

    output = build_ground_truth(catalogue_payload, scene_payload)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Seeded ground truth skeleton written to {output_path.resolve()}")


if __name__ == "__main__":
    raise SystemExit(main())
