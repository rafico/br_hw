from collections import defaultdict
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.optional_deps import require_modules

require_modules("cv2", "torch", "torchreid", "transformers")

import reid_ensemble
import run
from visualizers.common import render_tracking_overlay


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
    def test_default_yolo_weights_maps_model_family(self):
        self.assertEqual(run._default_yolo_weights("yolov11"), "yolo11m.pt")
        self.assertEqual(run._default_yolo_weights("yolov26"), "yolo26m.pt")

    def test_parse_args_accepts_visualizer_and_yolo_model(self):
        args = run.parse_args([
            "--dataset-dir", "/tmp/videos",
            "--visualizer", "rerun",
            "--yolo-model", "yolov11",
            "--rerun-save", "demo.rrd",
        ])

        self.assertEqual(args.dataset_dir, "/tmp/videos")
        self.assertEqual(args.visualizer, "rerun")
        self.assertEqual(args.yolo_model, "yolov11")
        self.assertEqual(args.rerun_save, "demo.rrd")

    def test_main_populates_default_yolo_weights_from_selected_family(self):
        captured = {}

        def fake_run_pipeline(args):
            captured["args"] = args

        with mock.patch.object(run._orchestrator, "run_pipeline", side_effect=fake_run_pipeline):
            run.main(["--dataset-dir", "/tmp/videos", "--yolo-model", "yolov11"])

        self.assertEqual(captured["args"].yolo_weights, "yolo11m.pt")

    def test_main_preserves_explicit_yolo_weights(self):
        captured = {}

        def fake_run_pipeline(args):
            captured["args"] = args

        with mock.patch.object(run._orchestrator, "run_pipeline", side_effect=fake_run_pipeline):
            run.main(["--dataset-dir", "/tmp/videos", "--yolo-model", "yolov11", "--yolo-weights", "custom.pt"])

        self.assertEqual(captured["args"].yolo_weights, "custom.pt")

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

    def test_run_pipeline_launches_fiftyone_app_for_fiftyone_visualizer(self):
        pipeline_orchestrator = run._orchestrator
        args = SimpleNamespace(
            finetune_reid=False,
            use_new_clustering=True,
            reid_backbone="ensemble",
            fo_dataset_name="re_id",
            dataset_dir="/tmp/videos",
            overwrite_loading=False,
            overwrite_algo=False,
            show=False,
            det_batch_size=8,
            yolo_weights="yolo26m.pt",
            reid_model_name="osnet_ain_x1_0",
            reid_weights="",
            reid_model_path="",
            tracker_type="botsort",
            tracker_reid_weights="",
            tracker_half=False,
            sim_key="embd_sim",
            viz_key="embd_viz",
            visualizer="fiftyone",
            pose_filter=False,
            scene_backend="gemini",
            rerun_save="qa_artifacts/recording.rrd",
            rerun_spawn=False,
            rerun_sample_every=1,
            rerun_max_frames_per_clip=None,
            evaluate=False,
            finetune_epochs=5,
            finetune_min_frames=30,
            finetune_min_prob=0.9,
        )
        dataset = mock.Mock()
        dataset.list_brain_runs.return_value = []
        frame_view = object()

        with mock.patch.object(pipeline_orchestrator, "seed_everything"), \
                mock.patch.object(pipeline_orchestrator, "time_stage", side_effect=lambda timings, stage, func: func()), \
                mock.patch.object(pipeline_orchestrator, "load_video_files", return_value=(dataset, False)), \
                mock.patch.object(pipeline_orchestrator, "process_video_file"), \
                mock.patch.object(pipeline_orchestrator, "get_frame_view", return_value=frame_view), \
                mock.patch.object(pipeline_orchestrator, "compute_similarity"), \
                mock.patch.object(pipeline_orchestrator, "compute_visualization"), \
                mock.patch.object(pipeline_orchestrator, "compute_or_load_all_detections", return_value=[]), \
                mock.patch.object(pipeline_orchestrator, "build_video_meta", return_value={}), \
                mock.patch.object(pipeline_orchestrator, "generate_person_catalogue_v2", return_value={"catalogue": {}}), \
                mock.patch.object(pipeline_orchestrator, "classify_scenes_vlm"), \
                mock.patch.object(pipeline_orchestrator, "write_timing_report"), \
                mock.patch.object(pipeline_orchestrator, "configure_dataset_visualization") as configure_mock, \
                mock.patch.object(pipeline_orchestrator, "launch_app") as launch_mock:
            pipeline_orchestrator.run_pipeline(args)

        configure_mock.assert_called_once_with(dataset)
        launch_mock.assert_called_once_with(frame_view)


class VisualizationTests(unittest.TestCase):
    def test_render_tracking_overlay_draws_tracks_on_a_copy(self):
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        tracks = np.array([[2.0, 2.0, 18.0, 18.0, 7.0, 0.91, 0.0, 0.0]], dtype=np.float32)

        rendered = render_tracking_overlay(frame, tracks, np.empty((0, 6), dtype=np.float32))

        self.assertEqual(rendered.shape, frame.shape)
        self.assertFalse(np.array_equal(rendered, frame))
        self.assertTrue(np.array_equal(frame, np.zeros_like(frame)))

    def test_render_tracking_overlay_falls_back_to_detector_boxes(self):
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        detections = np.array([[3.0, 3.0, 19.0, 19.0, 0.95, 0.0]], dtype=np.float32)

        rendered = render_tracking_overlay(frame, np.empty((0, 8), dtype=np.float32), detections)

        self.assertFalse(np.array_equal(rendered, frame))

    def test_process_frame_batch_shows_annotated_frame(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        annotated = np.ones_like(frame)
        sample = SimpleNamespace(frames=defaultdict(dict))
        model = mock.Mock(return_value=[SimpleNamespace(boxes=None)])
        tracker = object()
        vision = run._vision

        with mock.patch.object(vision, "update_tracker", return_value=np.empty((0, 8), dtype=np.float32)), \
                mock.patch.object(vision, "convert_to_fiftyone_detections", return_value=[]), \
                mock.patch.object(vision, "render_tracking_overlay", return_value=annotated) as render_mock, \
                mock.patch.object(vision.cv2, "imshow") as imshow_mock, \
                mock.patch.object(vision.cv2, "waitKey", return_value=-1), \
                mock.patch.object(vision.fo, "Detections", side_effect=lambda detections: SimpleNamespace(detections=detections)):
            stop_requested = vision._process_frame_batch(
                frames=[frame],
                frame_numbers=[1],
                fps=30.0,
                model=model,
                person_class_id=0,
                person_label="person",
                reid_extractor=mock.Mock(),
                tracker=tracker,
                sample=sample,
                width=16,
                height=16,
                show_visuals=True,
            )

        self.assertFalse(stop_requested)
        render_mock.assert_called_once()
        imshow_mock.assert_called_once_with("BoXMOT + Ultralytics", annotated)
        self.assertEqual(sample.frames[1]["detections"].detections, [])


if __name__ == "__main__":
    unittest.main()
