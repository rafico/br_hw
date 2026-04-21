"""Compatibility entrypoint for the pipeline.

The implementation now lives under the `pipeline` package so orchestration,
video processing, dataset IO, and re-embedding logic can evolve independently.
"""

from boxmot import BotSort, ByteTrack, DeepOcSort, StrongSort
from ultralytics import YOLO

from pipeline import cli as _cli
from pipeline import dataset as _dataset
from pipeline import orchestrator as _orchestrator
from pipeline import reembed as _reembed
from pipeline import timings as _timings
from pipeline import vision as _vision

_yolo_inference_kwargs = _vision._yolo_inference_kwargs
run_detection = _vision.run_detection
extract_reid_features = _vision.extract_reid_features
_compute_detection_quality = _vision._compute_detection_quality
convert_to_fiftyone_detections = _vision.convert_to_fiftyone_detections
_extract_person_detections = _vision._extract_person_detections
_process_frame_batch = _vision._process_frame_batch
process_single_video = _vision.process_single_video
configure_dataset_visualization = _dataset.configure_dataset_visualization
get_frame_view = _dataset.get_frame_view
compute_similarity = _dataset.compute_similarity
compute_visualization = _dataset.compute_visualization
launch_app = _dataset.launch_app
time_stage = _timings.time_stage
write_timing_report = _timings.write_timing_report
_normalize_rows = _reembed._normalize_rows
_load_rgb_frame_for_reembedding = _reembed._load_rgb_frame_for_reembedding
_merge_clip_embedding = _reembed._merge_clip_embedding
_read_json = _reembed.read_json
build_video_meta = _orchestrator.build_video_meta


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


def parse_args():
    return _cli.parse_args()


def main():
    args = parse_args()
    _orchestrator.run_pipeline(args)


if __name__ == "__main__":
    main()
