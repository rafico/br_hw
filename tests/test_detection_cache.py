import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

import compute_or_load_all_detections as cache_mod


class CacheKeyTests(unittest.TestCase):
    def test_cache_key_includes_reid_backbone(self):
        class DatasetStub(SimpleNamespace):
            def __len__(self):
                return 4

        dataset = DatasetStub(name="demo", media_type="video")

        key_osnet = cache_mod._cache_key_for_dataset(dataset, reid_backbone="osnet_ain")
        key_clip = cache_mod._cache_key_for_dataset(dataset, reid_backbone="clipreid")

        self.assertNotEqual(key_osnet, key_clip)

    def test_save_and_load_all_detections_round_trip(self):
        detections = [
            {
                "clip_id": "clip_1",
                "video_path": "/tmp/clip_1.mp4",
                "frame_num": 3,
                "track_id": 7,
                "embeddings": np.asarray([1.0, 0.0], dtype=np.float32),
                "torso_hist": np.asarray([0.5, 0.5], dtype=np.float32),
                "box_xyxy_abs": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = cache_mod.Path(tmpdir) / "all_detections.json"
            cache_mod.save_all_detections(path, detections)
            loaded = cache_mod.load_all_detections(path)

        self.assertEqual(loaded[0]["clip_id"], "clip_1")
        self.assertEqual(loaded[0]["embeddings"], [1.0, 0.0])
        self.assertEqual(loaded[0]["torso_hist"], [0.5, 0.5])
        self.assertEqual(loaded[0]["box_xyxy_abs"], [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
