"""Compatibility entrypoint for the pipeline.

The implementation now lives under the `pipeline` package so orchestration,
video processing, dataset IO, and re-embedding logic can evolve independently.
"""

import argparse
import importlib


class _LazyModule:
    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


def _load_boxmot_symbols():
    module = importlib.import_module("boxmot")
    return {
        "BotSort": module.BotSort,
        "ByteTrack": module.ByteTrack,
        "DeepOcSort": module.DeepOcSort,
        "StrongSort": module.StrongSort,
    }


def _load_yolo_cls():
    return importlib.import_module("ultralytics").YOLO


class YOLO:
    def __new__(cls, *args, **kwargs):
        return _load_yolo_cls()(*args, **kwargs)


class BotSort:
    def __new__(cls, *args, **kwargs):
        return _load_boxmot_symbols()["BotSort"](*args, **kwargs)


class ByteTrack:
    def __new__(cls, *args, **kwargs):
        return _load_boxmot_symbols()["ByteTrack"](*args, **kwargs)


class DeepOcSort:
    def __new__(cls, *args, **kwargs):
        return _load_boxmot_symbols()["DeepOcSort"](*args, **kwargs)


class StrongSort:
    def __new__(cls, *args, **kwargs):
        return _load_boxmot_symbols()["StrongSort"](*args, **kwargs)


_dataset = _LazyModule("pipeline.dataset")
_orchestrator = _LazyModule("pipeline.orchestrator")
_reembed = _LazyModule("pipeline.reembed")
_timings = _LazyModule("pipeline.timings")
_vision = _LazyModule("pipeline.vision")


def _default_yolo_weights(yolo_model: str = "yolov26", size: str = "m") -> str:
    normalized_model = str(yolo_model).lower()
    if normalized_model == "yolov11":
        return f"yolo11{size}.pt"
    if normalized_model == "yolov26":
        return f"yolo26{size}.pt"
    raise ValueError(f"Unsupported YOLO model family: {yolo_model!r}")


def load_detector(model_path: str = "yolo26m.pt"):
    return _vision.load_detector(model_path, yolo_cls=YOLO)


def load_reid_extractor(
        model_name: str = "osnet_ain_x1_0",
        model_path: str = "",
        image_size=(256, 128),
        batch_size: int = 32,
        input_is_bgr: bool = False,
        device: str | None = None,
):
    return _vision.load_reid_extractor(
        model_name=model_name,
        model_path=model_path,
        image_size=image_size,
        batch_size=batch_size,
        input_is_bgr=input_is_bgr,
        device=device,
    )


def _default_tracker_reid_weights():
    return _vision._default_tracker_reid_weights()


def load_tracker(
        tracker_type: str = "bytetrack",
        device: str = "cpu",
        tracker_reid_weights: str | None = None,
        tracker_half: bool = False,
):
    return _vision.load_tracker(
        tracker_type=tracker_type,
        device=device,
        tracker_reid_weights=tracker_reid_weights,
        tracker_half=tracker_half,
        bytetrack_cls=ByteTrack,
        botsort_cls=BotSort,
        strongsort_cls=StrongSort,
        deepocsort_cls=DeepOcSort,
    )


def _tracker_update_kw(tracker):
    return _vision._tracker_update_kw(tracker)


def _yolo_inference_kwargs():
    return _vision._yolo_inference_kwargs()


def run_detection(model, frame, person_class_id):
    return _vision.run_detection(model, frame, person_class_id)


def extract_reid_features(reid_extractor, rgb, boxes, detections):
    return _vision.extract_reid_features(reid_extractor, rgb, boxes, detections)


def _compute_detection_quality(rgb_frame, boxes, detections):
    return _vision._compute_detection_quality(rgb_frame, boxes, detections)


def convert_to_fiftyone_detections(detections, features, boxes, sharpness_scores, color_hists):
    return _vision.convert_to_fiftyone_detections(detections, features, boxes, sharpness_scores, color_hists)


def _extract_person_detections(result, person_class_id):
    return _vision._extract_person_detections(result, person_class_id)


def _process_frame_batch(*args, **kwargs):
    return _vision._process_frame_batch(*args, **kwargs)


def process_single_video(*args, **kwargs):
    return _vision.process_single_video(*args, **kwargs)


def configure_dataset_visualization(dataset):
    return _dataset.configure_dataset_visualization(dataset)


def get_frame_view(dataset):
    return _dataset.get_frame_view(dataset)


def compute_similarity(frame_view, sim_key):
    return _dataset.compute_similarity(frame_view, sim_key)


def compute_visualization(frame_view, sim_key, viz_key):
    return _dataset.compute_visualization(frame_view, sim_key, viz_key)


def launch_app(frame_view):
    return _dataset.launch_app(frame_view)


def time_stage(*args, **kwargs):
    return _timings.time_stage(*args, **kwargs)


def write_timing_report(*args, **kwargs):
    return _timings.write_timing_report(*args, **kwargs)


def _normalize_rows(arr):
    return _reembed._normalize_rows(arr)


def _load_rgb_frame_for_reembedding(video_path, frame_num):
    return _reembed._load_rgb_frame_for_reembedding(video_path, frame_num)


def _merge_clip_embedding(existing_embedding, clip_embedding, reid_backbone):
    return _reembed._merge_clip_embedding(existing_embedding, clip_embedding, reid_backbone)


def _read_json(path):
    return _reembed.read_json(path)


def build_video_meta(dataset):
    return _orchestrator.build_video_meta(dataset)


def update_tracker(tracker, detections, frame, features):
    return _vision.update_tracker(tracker, detections, frame, features)


def process_video_file(
        dataset,
        show_visuals: bool = False,
        det_batch_size: int = 8,
        yolo_weights: str = "yolo26m.pt",
        reid_model_name: str = "osnet_ain_x1_0",
        reid_backbone: str | None = None,
        reid_model_path: str = "",
        tracker_type: str = "bytetrack",
        tracker_reid_weights: str | None = None,
        tracker_half: bool = False,
):
    return _vision.process_video_file(
        dataset,
        show_visuals=show_visuals,
        det_batch_size=det_batch_size,
        yolo_weights=yolo_weights,
        reid_model_name=reid_model_name,
        reid_backbone=reid_backbone,
        reid_model_path=reid_model_path,
        tracker_type=tracker_type,
        tracker_reid_weights=tracker_reid_weights,
        tracker_half=tracker_half,
        detector_loader=load_detector,
        extractor_loader=load_reid_extractor,
        tracker_loader=load_tracker,
        single_video_processor=process_single_video,
    )


def load_video_files(fo_dataset_name, dataset_dir, overwrite):
    return _dataset.load_video_files(fo_dataset_name, dataset_dir, overwrite)


def reembed_detections_with_finetuned_clip(
        all_detections,
        *,
        reid_backbone: str,
        clip_weights_path: str,
        device: str | None = None,
):
    return _reembed.reembed_detections_with_finetuned_clip(
        all_detections,
        reid_backbone=reid_backbone,
        clip_weights_path=clip_weights_path,
        device=device,
        extractor_loader=load_reid_extractor,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create a FiftyOne dataset from a video directory.")
    parser.add_argument("--fo-dataset-name", default="re_id", help="Name of the dataset")
    parser.add_argument("--dataset-dir", required=True, help="Path to the videos directory")
    parser.add_argument("--yolo-model", default="yolov26", choices=["yolov11", "yolov26"], help="Detector family preset used for the default medium checkpoint")
    parser.add_argument("--yolo-weights", default="", help="Explicit detector weights path/name; overrides --yolo-model")
    parser.add_argument("--show", action="store_true", help="Show live video visualization during processing")
    parser.add_argument("--overwrite-loading", action="store_true", help="reload fo dataset")
    parser.add_argument("--overwrite-algo", action="store_true", help="recompute embedding")
    parser.add_argument("--save-visual-debug", action="store_true", help="Save cropped images of each person for visual debugging")
    parser.add_argument("--visual-debug-dir", default="debug_visual", help="Directory to save debug images")
    parser.add_argument("--sim-key", default="embd_sim", help="Brain key for similarity index")
    parser.add_argument("--viz-key", default="embd_viz", help="Brain key for visualization")
    parser.add_argument("--det-batch-size", type=int, default=8, help="Batch size for YOLO frame inference")
    parser.add_argument("--reid-model-name", default="osnet_ain_x1_0", help="Torchreid model name")
    parser.add_argument("--reid-backbone", default="ensemble", choices=["osnet_ain", "clipreid", "ensemble"], help="ReID feature extractor backend")
    parser.add_argument("--reid-model-path", default="", help="Optional fine-tuned ReID checkpoint path")
    parser.add_argument("--reid-weights", default="", help="Override CLIP ReID checkpoint path")
    parser.add_argument("--tracker-type", default="botsort", choices=["bytetrack", "botsort", "strongsort", "deepocsort"], help="Tracker backend")
    parser.add_argument("--tracker-reid-weights", default="", help="ReID weights for StrongSORT/DeepOCSORT (defaults to BoxMOT osnet_x0_25_msmt17.pt)")
    parser.add_argument("--tracker-half", action="store_true", help="Use half precision for tracker ReID model (CUDA only)")
    parser.add_argument("--disable-sparse-clustering", action="store_true", help="Disable sparse neighbor clustering path")
    parser.add_argument("--sparse-threshold", type=int, default=1000, help="Tracklet count threshold to enable sparse clustering")
    parser.add_argument("--linkage", default="min", choices=["min", "mean", "representative"], help="Tracklet linkage for distance computation")
    parser.add_argument("--min-tracklet-frames", type=int, default=2, help="Minimum frames per tracklet")
    parser.add_argument("--disable-quality-weighting", action="store_true", help="Disable confidence/sharpness weighted representative embeddings")
    parser.add_argument("--quality-alpha", type=float, default=0.75, help="Weight of quality scores in representative aggregation [0,1]")
    parser.add_argument("--disable-temporal-smoothing", action="store_true", help="Disable temporal smoothing of embeddings within each tracklet")
    parser.add_argument("--smoothing-window", type=int, default=5, help="Temporal smoothing window size (odd recommended)")
    parser.add_argument("--temporal-penalty", type=float, default=0.05, help="Soft penalty on temporal-center mismatch across clips")
    parser.add_argument("--temporal-max-gap-sec", type=float, default=None, help="Optional hard max gap (seconds) when absolute clip timestamps are parseable")
    parser.add_argument("--motion-penalty", type=float, default=0.05, help="Soft penalty on motion-profile mismatch")
    parser.add_argument("--disable-postprocess-merge", action="store_true", help="Disable post-clustering singleton merge pass")
    parser.add_argument("--postprocess-merge-epsilon", type=float, default=None, help="Distance threshold for post-clustering merges")
    parser.add_argument("--use-rerank", action="store_true", help="Use k-reciprocal re-ranking on representative embeddings")
    parser.add_argument("--cooccurrence-constraint", action="store_true", help="Allow same-clip merges only when tracklets do not overlap")
    parser.set_defaults(use_new_clustering=True)
    parser.add_argument("--use-new-clustering", dest="use_new_clustering", action="store_true", help="Use the HDBSCAN multi-prototype clustering pipeline")
    parser.add_argument("--legacy-clustering", dest="use_new_clustering", action="store_false", help="Use the legacy catalogue_simple clustering pipeline")
    parser.add_argument("--pose-filter", action="store_true", help="Enable MediaPipe pose filtering for V2 keyframe selection")
    parser.add_argument("--scene-backend", default="gemini", choices=["gemini", "videomae", "internvideo"], help="Scene classification backend")
    parser.add_argument("--visualizer", default="fiftyone", choices=["none", "fiftyone", "rerun", "both"], help="Post-run visualization backend")
    parser.add_argument("--rerun-spawn", action="store_true", help="Spawn a Rerun viewer after export")
    parser.add_argument("--rerun-save", default="", help="Optional path to save a Rerun .rrd recording")
    parser.add_argument("--rerun-sample-every", type=int, default=1, help="Sample every Nth frame when exporting frames to Rerun")
    parser.add_argument("--rerun-max-frames-per-clip", type=int, default=None, help="Optional cap on exported frame images per clip")
    parser.add_argument("--finetune-reid", action="store_true", help="Run a self-bootstrap CLIP fine-tune pass before the final clustering pass")
    parser.add_argument("--finetune-min-frames", type=int, default=30, help="Minimum total frames across appearances for pseudo-label fine-tuning")
    parser.add_argument("--finetune-min-prob", type=float, default=0.9, help="Minimum cluster_probability gate for pseudo-label fine-tuning")
    parser.add_argument("--finetune-epochs", type=int, default=5, help="Epoch count for pseudo-label CLIP fine-tuning")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation against ground_truth.json if available")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.yolo_weights:
        args.yolo_weights = _default_yolo_weights(args.yolo_model)
    _orchestrator.run_pipeline(args)


if __name__ == "__main__":
    main()
