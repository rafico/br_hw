import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from visualizers.projection import project_2d
from visualizers.rerun_export import _color_for_id, export_to_rerun


class _FakeRerun:
    Box2DFormat = types.SimpleNamespace(XYXY="XYXY")

    def __init__(self):
        self.logs = []
        self.saved_path = None
        self.spawned = False
        self.init_calls = []
        self.time_sequence = []
        self.time_seconds = []

    def init(self, app_id):
        self.init_calls.append(app_id)

    def save(self, path):
        self.saved_path = path

    def spawn(self):
        self.spawned = True

    def set_time_sequence(self, timeline, sequence):
        self.time_sequence.append((timeline, sequence))

    def set_time_seconds(self, timeline, seconds):
        self.time_seconds.append((timeline, seconds))

    def log(self, path, payload):
        self.logs.append((path, payload))

    def Scalar(self, value):
        return {"kind": "Scalar", "value": value}

    def TextDocument(self, text):
        return {"kind": "TextDocument", "text": text}

    def Image(self, image):
        return {"kind": "Image", "shape": tuple(image.shape)}

    def Boxes2D(self, **kwargs):
        return {"kind": "Boxes2D", "kwargs": kwargs}

    def Points2D(self, positions, **kwargs):
        return {
            "kind": "Points2D",
            "positions": np.asarray(positions, dtype=np.float32),
            "kwargs": kwargs,
        }


class ProjectionTests(unittest.TestCase):
    def test_project_2d_is_deterministic(self):
        rng = np.random.default_rng(51)
        features = rng.normal(size=(8, 6)).astype(np.float32)

        coords_a = project_2d(features, seed=51)
        coords_b = project_2d(features, seed=51)

        self.assertEqual(coords_a.shape, (8, 2))
        self.assertTrue(np.allclose(coords_a, coords_b))


class RerunExporterTests(unittest.TestCase):
    def test_exporter_logs_expected_paths_and_colors_by_global_id(self):
        fake_rr = _FakeRerun()
        detections = [
            {
                "clip_id": "clip_a",
                "video_path": "/tmp/clip_a.mp4",
                "frame_num": 1,
                "track_id": 7,
                "embeddings": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "timestamp_sec": 0.0,
                "box_xyxy_abs": [1.0, 2.0, 11.0, 22.0],
            },
            {
                "clip_id": "clip_b",
                "video_path": "/tmp/clip_b.mp4",
                "frame_num": 2,
                "track_id": 9,
                "embeddings": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "timestamp_sec": 0.2,
                "box_xyxy_abs": [3.0, 4.0, 13.0, 24.0],
            },
        ]
        catalogue_payload = {
            "summary": {
                "total_unique_persons": 1,
                "total_tracklets": 2,
                "adaptive": {"epsilon": 0.18, "gate": 0.2},
            },
            "catalogue": {
                "1": [
                    {
                        "clip_id": "clip_a",
                        "local_track_id": 7,
                        "cluster_probability": 0.97,
                        "frame_ranges": [[1, 4]],
                    }
                ]
            },
        }
        scene_payload = [
            {
                "clip_id": "clip_a",
                "label": "crime",
                "crime_segments": [
                    {
                        "timestamp_start": 0.1,
                        "timestamp_end": 0.9,
                        "involved_people_global": [1],
                    }
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch("visualizers.rerun_export._load_rerun", return_value=fake_rr), \
                mock.patch("visualizers.rerun_export._load_rgb_frame", return_value=np.zeros((8, 8, 3), dtype=np.uint8)):
            output_path = Path(tmpdir) / "recording.rrd"
            export_to_rerun(
                detections=detections,
                catalogue_payload=catalogue_payload,
                scene_payload=scene_payload,
                output_rrd=str(output_path),
                sample_every=1,
                max_frames_per_clip=1,
            )

        paths = {path for path, _ in fake_rr.logs}
        self.assertIn("metrics/total_unique_persons", paths)
        self.assertIn("clips/clip_a/frame", paths)
        self.assertIn("clips/clip_a/detections", paths)
        self.assertIn("embeddings/detections_2d", paths)
        self.assertIn("embeddings/tracklets_2d", paths)
        self.assertIn("scenes/clip_a/summary", paths)
        self.assertEqual(fake_rr.saved_path, str(output_path))

        clip_boxes = dict(fake_rr.logs)["clips/clip_a/detections"]
        self.assertEqual(
            clip_boxes["kwargs"]["colors"][0],
            _color_for_id(1, seed=51),
        )

    def test_missing_dependency_raises_clear_error(self):
        with mock.patch("importlib.import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(RuntimeError, "requirements-rerun.txt"):
                export_to_rerun(
                    detections=[],
                    catalogue_payload={},
                    scene_payload=[],
                )


if __name__ == "__main__":
    unittest.main()
