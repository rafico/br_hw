import tempfile
import unittest
from pathlib import Path

from qa.manual_review import build_review_manifest, resolve_detection_cache_path
from qa.output_validation import (
    validate_catalogue_payload,
    validate_eval_report_payload,
    validate_scene_payload,
)
from qa.runner import build_suite_commands


class OutputValidationTests(unittest.TestCase):
    def test_validate_catalogue_payload_accepts_valid_payload(self):
        payload = {
            "catalogue": {
                "1": [
                    {
                        "clip_id": "clip_1",
                        "local_track_id": 7,
                        "frame_ranges": [[1, 3], [7, 9]],
                        "cluster_probability": 0.95,
                    }
                ]
            }
        }

        self.assertEqual(validate_catalogue_payload(payload), [])

    def test_validate_catalogue_payload_rejects_bad_probability_and_ranges(self):
        payload = {
            "catalogue": {
                "a": [
                    {
                        "clip_id": "clip_1",
                        "local_track_id": 7,
                        "frame_ranges": [[5, 3]],
                        "cluster_probability": 2.0,
                    }
                ]
            }
        }

        errors = validate_catalogue_payload(payload)
        self.assertTrue(any("int-like string" in error for error in errors))
        self.assertTrue(any("cluster_probability" in error for error in errors))
        self.assertTrue(any("start <=" in error for error in errors))

    def test_validate_scene_payload_rejects_invalid_segments(self):
        payload = [
            {
                "clip_id": "clip_1",
                "label": "crime",
                "crime_segments": [
                    {
                        "timestamp_start": 5.0,
                        "timestamp_end": 2.0,
                        "involved_people_global": ["7"],
                    }
                ],
            }
        ]

        errors = validate_scene_payload(payload)
        self.assertTrue(any("timestamp_start <=" in error for error in errors))
        self.assertTrue(any("must be an int" in error for error in errors))

    def test_validate_eval_report_payload_checks_metric_bounds(self):
        payload = {
            "person_reid": {
                "v_measure": 1.2,
                "adjusted_rand_index": 0.5,
                "purity": 1.0,
            },
            "scene": {
                "accuracy": -0.1,
                "macro_f1": 0.8,
            },
        }

        errors = validate_eval_report_payload(payload)
        self.assertTrue(any("v_measure" in error for error in errors))
        self.assertTrue(any("scene.accuracy" in error for error in errors))


class RunnerTests(unittest.TestCase):
    def test_build_suite_commands_for_offline_contains_help_and_unit_steps(self):
        commands = build_suite_commands("offline", python_bin="/usr/bin/python3")
        names = [command.name for command in commands]

        self.assertIn("unit_tests", names)
        self.assertIn("run_help", names)
        self.assertIn("qa_validate_help", names)

    def test_build_suite_commands_for_dataset_suite_requires_dataset_dir(self):
        with self.assertRaises(ValueError):
            build_suite_commands("dataset-smoke", python_bin="/usr/bin/python3")

    def test_build_suite_commands_for_release_local_includes_rerun_and_validation(self):
        commands = build_suite_commands(
            "release-local",
            python_bin="/usr/bin/python3",
            dataset_dir="/tmp/videos",
            rerun_save="qa_recording.rrd",
        )
        names = [command.name for command in commands]

        self.assertIn("legacy_pipeline_smoke", names)
        self.assertIn("finetune_release_path", names)
        self.assertIn("rerun_release_path", names)
        rerun_command = next(command for command in commands if command.name == "rerun_release_path")
        self.assertIn("qa_recording.rrd", rerun_command.argv)

    def test_build_suite_commands_for_manual_visual_prep_includes_bundle_generation(self):
        commands = build_suite_commands(
            "manual-visual-prep",
            python_bin="/usr/bin/python3",
            dataset_dir="/tmp/videos",
            rerun_save="qa_artifacts/recording.rrd",
        )
        names = [command.name for command in commands]

        self.assertEqual(
            names,
            [
                "manual_visual_rerun_export",
                "manual_visual_output_validation",
                "manual_visual_review_bundle",
            ],
        )
        bundle_command = next(command for command in commands if command.name == "manual_visual_review_bundle")
        self.assertIn("scripts/prepare_manual_visual_review.py", bundle_command.argv)
        self.assertIn(str(Path.cwd() / "qa_artifacts" / "manual_visual_review.json"), bundle_command.argv)


class ManualReviewTests(unittest.TestCase):
    def test_resolve_detection_cache_path_prefers_newest_variant_when_base_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache"
            cache_dir.mkdir()
            older = cache_dir / "demo_all_detections_ft.json"
            newer = cache_dir / "demo_all_detections_debug.json"
            older.write_text("[]", encoding="utf-8")
            newer.write_text("[]", encoding="utf-8")
            newer.touch()

            resolved = resolve_detection_cache_path(tmpdir)

        self.assertEqual(resolved.name, "demo_all_detections_debug.json")

    def test_resolve_detection_cache_path_prefers_base_cache_over_variants(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache"
            cache_dir.mkdir()
            base = cache_dir / "demo_all_detections.json"
            ft = cache_dir / "demo_all_detections_ft.json"
            base.write_text("[]", encoding="utf-8")
            ft.write_text("[]", encoding="utf-8")
            ft.touch()

            resolved = resolve_detection_cache_path(tmpdir)

        self.assertEqual(resolved.name, "demo_all_detections.json")

    def test_build_review_manifest_links_catalogue_and_scene_metadata(self):
        detections = [
            {
                "clip_id": "clip_1",
                "frame_num": 1,
                "track_id": 7,
                "timestamp_sec": 0.0,
                "confidence": 0.9,
                "box_xyxy_abs": [1.0, 2.0, 3.0, 4.0],
            },
            {
                "clip_id": "clip_1",
                "frame_num": 5,
                "track_id": 7,
                "timestamp_sec": 0.4,
                "confidence": 0.8,
                "box_xyxy_abs": [2.0, 3.0, 4.0, 5.0],
            },
            {
                "clip_id": "clip_2",
                "frame_num": 3,
                "track_id": 9,
                "timestamp_sec": 0.2,
                "confidence": 0.95,
                "box_xyxy_abs": [5.0, 6.0, 7.0, 8.0],
            },
        ]
        catalogue_payload = {
            "catalogue": {
                "1": [
                    {
                        "clip_id": "clip_1",
                        "local_track_id": 7,
                        "frame_ranges": [[1, 6]],
                        "cluster_probability": 0.92,
                    }
                ],
                "2": [
                    {
                        "clip_id": "clip_2",
                        "local_track_id": 9,
                        "frame_ranges": [[3, 4]],
                        "cluster_probability": 0.88,
                    }
                ],
            }
        }
        scene_payload = [
            {
                "clip_id": "clip_1",
                "label": "crime",
                "crime_segments": [
                    {
                        "timestamp_start": 0.0,
                        "timestamp_end": 0.5,
                        "involved_people_global": [1],
                    }
                ],
            },
            {
                "clip_id": "clip_2",
                "label": "normal",
                "crime_segments": [],
            },
        ]

        manifest = build_review_manifest(
            dataset_dir="/tmp/videos",
            detections=detections,
            catalogue_payload=catalogue_payload,
            scene_payload=scene_payload,
            detections_cache_path="/tmp/videos/.cache/demo_all_detections.json",
            catalogue_path="catalogue_v2.json",
            scene_path="scene_labels_v2.json",
            rerun_recording_path="qa_artifacts/recording.rrd",
            sample_count=3,
            consistency_count=2,
        )

        self.assertEqual(manifest["summary"]["review_frame_count"], 3)
        self.assertEqual(manifest["summary"]["clip_count"], 2)
        self.assertEqual(len(manifest["consistency_frames"]), 2)
        first_clip = next(entry for entry in manifest["review_frames"] if entry["clip_id"] == "clip_1")
        self.assertEqual(first_clip["detections"][0]["global_id"], 1)
        self.assertTrue(first_clip["detections"][0]["catalogue_frame_match"])
        self.assertEqual(first_clip["scene_segments_overlapping_frame"][0]["involved_people_global"], [1])
        self.assertEqual(manifest["warnings"], [])


if __name__ == "__main__":
    unittest.main()
