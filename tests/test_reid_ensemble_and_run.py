import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import compute_or_load_all_detections as cache_mod
import reid_ensemble
import run


class StubExtractor:
    def __init__(self, feats, keep_idx):
        self.feats = np.asarray(feats, dtype=np.float32)
        self.keep_idx = np.asarray(keep_idx, dtype=np.int64)

    def extract_from_detections(self, frame, boxes_xyxy):
        return self.feats.copy(), self.keep_idx.copy()


class EnsembleExtractorTests(unittest.TestCase):
    def test_ensemble_extractor_intersects_keep_indices_and_renormalizes(self):
        extractor = reid_ensemble.EnsembleExtractor(
            [
                StubExtractor([[1.0, 0.0], [0.0, 1.0]], [0, 2]),
                StubExtractor([[0.5, 0.5], [1.0, 0.0]], [2, 3]),
            ]
        )

        feats, keep_idx = extractor.extract_from_detections(
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.float32),
        )

        self.assertTrue(np.array_equal(keep_idx, np.array([2], dtype=np.int64)))
        self.assertEqual(feats.shape, (1, 4))
        self.assertTrue(np.allclose(np.linalg.norm(feats, axis=1), 1.0))

    def test_build_extractor_routes_to_requested_backend(self):
        with mock.patch.object(reid_ensemble, "DetectionReIDExtractor", autospec=True) as osnet_cls, \
                mock.patch.object(reid_ensemble, "CLIPReIDExtractor", autospec=True) as clip_cls:
            osnet_instance = object()
            clip_instance = object()
            osnet_cls.return_value = osnet_instance
            clip_cls.return_value = clip_instance

            self.assertIs(
                reid_ensemble.build_extractor("osnet_ain", device="cpu"),
                osnet_instance,
            )
            self.assertIs(
                reid_ensemble.build_extractor("clipreid", device="cpu"),
                clip_instance,
            )
            ensemble = reid_ensemble.build_extractor("ensemble", device="cpu")

        self.assertIsInstance(ensemble, reid_ensemble.EnsembleExtractor)
        self.assertEqual(len(ensemble.extractors), 2)


class RunTrackerTests(unittest.TestCase):
    def test_load_detector_falls_back_from_yolo26_to_yolo11_when_missing(self):
        class FakeModel:
            names = {0: "person"}

        def fake_yolo(weights):
            if weights == "yolo26m.pt":
                raise FileNotFoundError("missing yolo26m.pt")
            if weights == "yolo11m.pt":
                return FakeModel()
            raise AssertionError(f"unexpected weights: {weights}")

        with mock.patch.object(run, "YOLO", side_effect=fake_yolo):
            model, person_class_id = run.load_detector("yolo26m.pt")

        self.assertIsInstance(model, FakeModel)
        self.assertEqual(person_class_id, 0)

    def test_load_tracker_supports_botsort(self):
        with mock.patch.object(run, "BotSort", autospec=True) as bot_sort:
            sentinel = object()
            bot_sort.return_value = sentinel

            tracker = run.load_tracker("botsort", device="cpu")

        self.assertIs(tracker, sentinel)
        kwargs = bot_sort.call_args.kwargs
        self.assertTrue(kwargs["with_reid"])
        self.assertEqual(kwargs["track_high_thresh"], 0.45)
        self.assertEqual(kwargs["appearance_thresh"], 0.25)

    def test_update_tracker_prefers_embs_keyword_when_supported(self):
        tracker = mock.Mock()

        def update(dets, frame, embs=None):
            return "ok"

        tracker.update = update
        result = run.update_tracker(tracker, "dets", "frame", "features")

        self.assertEqual(result, "ok")

    def test_update_tracker_falls_back_to_positional_features(self):
        tracker = mock.Mock()

        def update(dets, frame, features):
            return (dets, frame, features)

        tracker.update = update
        result = run.update_tracker(tracker, "dets", "frame", "features")

        self.assertEqual(result, ("dets", "frame", "features"))


class CacheKeyTests(unittest.TestCase):
    def test_cache_key_includes_reid_backbone(self):
        class DatasetStub(SimpleNamespace):
            def __len__(self):
                return 4

        dataset = DatasetStub(name="demo", media_type="video")

        key_osnet = cache_mod._cache_key_for_dataset(dataset, reid_backbone="osnet_ain")
        key_clip = cache_mod._cache_key_for_dataset(dataset, reid_backbone="clipreid")

        self.assertNotEqual(key_osnet, key_clip)


if __name__ == "__main__":
    unittest.main()
