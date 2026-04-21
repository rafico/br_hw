from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_ground_truth(catalogue_payload: dict, scene_payload: list[dict]) -> dict:
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
                if isinstance(person_id, int)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Seed ground_truth.json from current predictions")
    parser.add_argument("--catalogue", default="catalogue_v2.json")
    parser.add_argument("--scenes", default="scene_labels_v2.json")
    parser.add_argument("--output", default="ground_truth.json")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.catalogue, "r", encoding="utf-8") as f:
        catalogue_payload = json.load(f)
    with open(args.scenes, "r", encoding="utf-8") as f:
        scene_payload = json.load(f)

    output = build_ground_truth(catalogue_payload, scene_payload)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Seeded ground truth skeleton written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
