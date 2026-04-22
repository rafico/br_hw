import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import evaluate
import vlm_scene


class VLMSceneTests(unittest.TestCase):
    def test_build_person_presence_and_resolve_persons_for_event(self):
        catalogue = {
            "1": [{"clip_id": "4", "frame_ranges": [[1, 10]]}],
            "3": [{"clip_id": "4", "frame_ranges": [[20, 30]]}],
        }
        presence = vlm_scene.build_person_presence(catalogue, {"4": 10.0})

        resolved = vlm_scene.resolve_persons_for_event(
            {"t_start_sec": 0.0, "t_end_sec": 0.9, "global_person_ids": [1, 9]},
            presence,
            "4",
        )

        self.assertEqual(resolved, [1])

    def test_classify_scenes_vlm_routes_backend_and_writes_output(self):
        sample = SimpleNamespace(
            filepath="/tmp/4.mp4",
            metadata=SimpleNamespace(frame_rate=10.0),
        )
        dataset = [sample]
        catalogue_payload = {"catalogue": {"1": [{"clip_id": "4", "frame_ranges": [[1, 10]]}]}}

        with tempfile.TemporaryDirectory() as tmpdir:
            catalogue_path = Path(tmpdir) / "catalogue_v2.json"
            output_path = Path(tmpdir) / "scene_labels_v2.json"
            catalogue_path.write_text(json.dumps(catalogue_payload), encoding="utf-8")

            with mock.patch.object(
                vlm_scene,
                "_call_gemini",
                return_value={
                    "label": "crime",
                    "confidence": 0.9,
                    "events": [
                        {
                            "type": "assault",
                            "t_start_sec": 0.1,
                            "t_end_sec": 0.8,
                            "global_person_ids": [],
                            "evidence": "fight",
                        }
                    ],
                    "rationale": "Observed a physical altercation.",
                },
            ):
                results = vlm_scene.classify_scenes_vlm(
                    dataset,
                    catalogue_path=str(catalogue_path),
                    output_file=str(output_path),
                    backend="gemini",
                )
                output_exists = output_path.exists()

        self.assertTrue(output_exists)
        self.assertEqual(results[0]["label"], "crime")
        self.assertEqual(results[0]["crime_segments"][0]["involved_people_global"], [1])

    def test_classify_scenes_vlm_normalizes_normal_label_with_events_to_crime(self):
        sample = SimpleNamespace(
            filepath="/tmp/4.mp4",
            metadata=SimpleNamespace(frame_rate=10.0),
        )
        dataset = [sample]
        catalogue_payload = {"catalogue": {"1": [{"clip_id": "4", "frame_ranges": [[1, 10]]}]}}

        with tempfile.TemporaryDirectory() as tmpdir:
            catalogue_path = Path(tmpdir) / "catalogue_v2.json"
            output_path = Path(tmpdir) / "scene_labels_v2.json"
            catalogue_path.write_text(json.dumps(catalogue_payload), encoding="utf-8")

            with mock.patch.object(
                vlm_scene,
                "_call_gemini",
                return_value={
                    "label": "normal",
                    "confidence": 0.9,
                    "events": [
                        {
                            "type": "assault",
                            "t_start_sec": 0.1,
                            "t_end_sec": 0.8,
                            "global_person_ids": [],
                            "evidence": "fight",
                        }
                    ],
                    "rationale": "Observed a physical altercation.",
                },
            ), self.assertLogs(vlm_scene.LOGGER, level="WARNING") as logs:
                results = vlm_scene.classify_scenes_vlm(
                    dataset,
                    catalogue_path=str(catalogue_path),
                    output_file=str(output_path),
                    backend="gemini",
                )

        self.assertEqual(results[0]["label"], "crime")
        self.assertTrue(results[0]["crime_segments"])
        self.assertIn("assault", results[0]["justification"])
        self.assertTrue(any("Normalized scene label" in message for message in logs.output))

    def test_classify_scenes_vlm_normalizes_crime_label_without_events_to_normal(self):
        sample = SimpleNamespace(
            filepath="/tmp/4.mp4",
            metadata=SimpleNamespace(frame_rate=10.0),
        )
        dataset = [sample]
        catalogue_payload = {"catalogue": {"1": [{"clip_id": "4", "frame_ranges": [[1, 10]]}]}}

        with tempfile.TemporaryDirectory() as tmpdir:
            catalogue_path = Path(tmpdir) / "catalogue_v2.json"
            output_path = Path(tmpdir) / "scene_labels_v2.json"
            catalogue_path.write_text(json.dumps(catalogue_payload), encoding="utf-8")

            with mock.patch.object(
                vlm_scene,
                "_call_gemini",
                return_value={
                    "label": "crime",
                    "confidence": 0.9,
                    "events": [],
                    "rationale": "No crime evidence remained after review.",
                },
            ), self.assertLogs(vlm_scene.LOGGER, level="WARNING") as logs:
                results = vlm_scene.classify_scenes_vlm(
                    dataset,
                    catalogue_path=str(catalogue_path),
                    output_file=str(output_path),
                    backend="gemini",
                )

        self.assertEqual(results[0]["label"], "normal")
        self.assertEqual(results[0]["crime_segments"], [])
        self.assertEqual(results[0]["justification"], "No criminal activity detected in the clip.")
        self.assertTrue(any("Normalized scene label" in message for message in logs.output))


class EvaluateTests(unittest.TestCase):
    def test_run_returns_none_when_ground_truth_missing(self):
        self.assertIsNone(evaluate.run(gt_path="does_not_exist.json"))

    def test_run_generates_report_for_perfect_prediction(self):
        gt = {
            "persons": [
                {"global_id": 1, "appearances": [{"clip": "1", "frame_ranges": [[1, 3]]}]}
            ],
            "scenes": [
                {"clip": "1", "label": "normal"}
            ],
        }
        pred_catalogue = {
            "catalogue": {
                "1": [{"clip_id": "1", "local_track_id": 7, "frame_ranges": [[1, 3]]}]
            }
        }
        pred_scene = [
            {"clip_id": "1", "label": "normal", "crime_segments": []}
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            gt_path = Path(tmpdir) / "ground_truth.json"
            pred_catalogue_path = Path(tmpdir) / "catalogue_v2.json"
            pred_scene_path = Path(tmpdir) / "scene_labels_v2.json"
            output_path = Path(tmpdir) / "eval_report.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8")
            pred_catalogue_path.write_text(json.dumps(pred_catalogue), encoding="utf-8")
            pred_scene_path.write_text(json.dumps(pred_scene), encoding="utf-8")

            report = evaluate.run(
                gt_path=str(gt_path),
                pred_catalogue=str(pred_catalogue_path),
                pred_scene=str(pred_scene_path),
                output=str(output_path),
            )
            output_exists = output_path.exists()
            md_exists = output_path.with_suffix(".md").exists()

            self.assertTrue(output_exists)
            self.assertTrue(md_exists)

        self.assertEqual(report["scene"]["accuracy"], 1.0)
        self.assertEqual(report["person_reid"]["purity"], 1.0)


if __name__ == "__main__":
    unittest.main()
