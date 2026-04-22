import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.optional_deps import require_modules

require_modules("cv2", "torch", "torchreid", "transformers")

import cluster_v2
from reid_ensemble import filter_crops_for_reid
from reid_model import torso_color_chi2, torso_color_hist


class ColorAndFilterTests(unittest.TestCase):
    def test_torso_color_hist_is_normalized_and_96d(self):
        frame = np.zeros((40, 20, 3), dtype=np.uint8)
        frame[:, :] = [255, 0, 0]

        hist = torso_color_hist(frame, np.array([2, 2, 18, 38], dtype=np.float32))

        self.assertEqual(hist.shape, (96,))
        self.assertAlmostEqual(float(hist.sum()), 1.0, places=5)

    def test_torso_color_chi2_is_symmetric(self):
        a = np.array([0.5, 0.5], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)

        self.assertAlmostEqual(torso_color_chi2(a, b), torso_color_chi2(b, a), places=6)

    def test_filter_crops_for_reid_applies_geometry_iou_and_pose_rules(self):
        boxes = np.array(
            [
                [0, 0, 10, 30],
                [0, 0, 30, 10],
                [0, 0, 20, 35],
            ],
            dtype=np.float32,
        )
        other = np.array([[2, 2, 18, 32]], dtype=np.float32)
        visible = lambda box: 8
        invisible = lambda box: 3

        keep_visible = filter_crops_for_reid(
            boxes_xyxy=boxes,
            frame_shape=(100, 100),
            other_boxes=other,
            use_pose=True,
            pose_model=visible,
        )
        keep_invisible = filter_crops_for_reid(
            boxes_xyxy=boxes[[2]],
            frame_shape=(100, 100),
            other_boxes=np.empty((0, 4), dtype=np.float32),
            use_pose=True,
            pose_model=invisible,
        )

        self.assertTrue(np.array_equal(keep_visible, np.array([True, False, False])))
        self.assertFalse(bool(keep_invisible[0]))


class ClusterV2Tests(unittest.TestCase):
    def _make_detection(self, clip_id, track_id, frame_num, emb, quality, torso_hist):
        return {
            "clip_id": str(clip_id),
            "video_path": f"/tmp/{clip_id}.mp4",
            "frame_num": frame_num,
            "track_id": track_id,
            "embeddings": np.asarray(emb, dtype=np.float32),
            "quality": quality,
            "timestamp_sec": frame_num / 10.0,
            "box_xyxy_abs": [0.0, 0.0, 10.0, 30.0],
            "frame_width": 100,
            "frame_height": 100,
            "torso_hist": np.asarray(torso_hist, dtype=np.float32),
            "center_x": 0.1,
            "center_y": 0.2,
            "bbox_w": 0.1,
            "bbox_h": 0.3,
        }

    def test_generate_person_catalogue_v2_writes_summary_and_catalogue(self):
        class FakeKMedoids:
            def __init__(self, n_clusters, metric, random_state):
                self.n_clusters = n_clusters

            def fit(self, embeddings):
                self.medoid_indices_ = np.arange(self.n_clusters)
                return self

        class FakeHDBSCAN:
            def __init__(self, **kwargs):
                pass

            def fit(self, dist_matrix):
                self.labels_ = np.array([0, -1], dtype=np.int32)
                self.probabilities_ = np.array([0.99, 0.2], dtype=np.float32)
                return self

        detections = [
            self._make_detection("1", 1, 1, [1.0, 0.0, 0.0], 0.9, [1.0, 0.0]),
            self._make_detection("1", 1, 2, [0.99, 0.01, 0.0], 0.8, [1.0, 0.0]),
            self._make_detection("2", 2, 1, [0.98, 0.02, 0.0], 0.95, [1.0, 0.0]),
            self._make_detection("2", 2, 2, [0.97, 0.03, 0.0], 0.85, [1.0, 0.0]),
        ]
        video_meta = {
            "1": {"fps": 10.0, "frame_count": 100},
            "2": {"fps": 10.0, "frame_count": 100},
        }

        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch.object(cluster_v2, "_get_clustering_backends", return_value=(FakeKMedoids, FakeHDBSCAN)):
            output_path = Path(tmpdir) / "catalogue_v2.json"
            result = cluster_v2.generate_person_catalogue_v2(
                detections,
                video_meta=video_meta,
                output_file=str(output_path),
                top_k_frames=2,
                n_prototypes=1,
                use_rerank=False,
                cooccurrence_constraint=True,
            )

            self.assertTrue(output_path.exists())
            payload = json.loads(output_path.read_text())

        self.assertIn("summary", result)
        self.assertIn("adaptive", result["summary"])
        self.assertEqual(payload["summary"]["total_unique_persons"], 1)
        self.assertEqual(sorted(payload["catalogue"].keys()), ["1"])


if __name__ == "__main__":
    unittest.main()
