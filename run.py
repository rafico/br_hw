import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import fiftyone as fo
import fiftyone.brain as fob
import numpy as np
import torch
from boxmot import ByteTrack, DeepOcSort, StrongSort
from boxmot.utils import WEIGHTS as BOXMOT_WEIGHTS
from ultralytics import YOLO

from reid_model import DetectionReIDExtractor
from generate_person_catalogue import generate_person_catalogue
from compute_or_load_all_detections import compute_or_load_all_detections
from classify_scenes import classify_scenes
from utils_determinism import seed_everything


def load_detector(model_path: str = "yolo11m.pt"):
    """Return (ultralytics YOLO model, person_class_id)."""
    model = YOLO(model_path)
    person_class_id = next((k for k, v in model.names.items() if v == "person"), None)
    return model, person_class_id


def load_reid_extractor(
        model_name: str = "osnet_ain_x1_0",
        model_path: str = "",
        image_size=(256, 128),
        batch_size: int = 32,
        input_is_bgr: bool = False,
        device: str | None = None,
):
    """Initialize and return the ReID extractor."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return DetectionReIDExtractor(
        model_name=model_name,
        model_path=model_path,
        image_size=image_size,
        device=device,
        batch_size=batch_size,
        input_is_bgr=input_is_bgr,
    )


def _default_tracker_reid_weights() -> Path:
    return BOXMOT_WEIGHTS / "osnet_x0_25_msmt17.pt"


def load_tracker(
        tracker_type: str = "bytetrack",
        device: str = "cpu",
        tracker_reid_weights: Optional[str] = None,
        tracker_half: bool = False,
):
    """Initialize a fresh tracker for each video to avoid cross-clip bleed-through.

    Improved settings to reduce track fragmentation:
    - max_age=60: keep lost tracks alive longer (default 30)
    - min_hits=1: confirm tracks faster (default 3)
    - det_thresh=0.2: lower detection threshold (default 0.3)
    """
    tracker_type = tracker_type.lower()
    tracker_common_args = {
        "max_age": 60,
        "min_hits": 1,
        "det_thresh": 0.2,
    }

    if tracker_type == "bytetrack":
        return ByteTrack(**tracker_common_args)

    tracker_device = torch.device(device)
    use_half = bool(tracker_half and tracker_device.type == "cuda")
    reid_weights = Path(tracker_reid_weights) if tracker_reid_weights else _default_tracker_reid_weights()

    try:
        if tracker_type == "strongsort":
            return StrongSort(
                reid_weights=reid_weights,
                device=tracker_device,
                half=use_half,
                **tracker_common_args,
            )
        if tracker_type == "deepocsort":
            return DeepOcSort(
                reid_weights=reid_weights,
                device=tracker_device,
                half=use_half,
                **tracker_common_args,
            )
    except Exception as exc:
        print(
            f"Tracker '{tracker_type}' initialization failed ({exc}). "
            "Falling back to ByteTrack."
        )
        return ByteTrack(**tracker_common_args)

    raise ValueError(
        f"Unsupported tracker_type '{tracker_type}'. "
        "Expected one of: bytetrack, strongsort, deepocsort."
    )


def run_detection(model, frame, person_class_id):
    """Run YOLO detection and filter for person class.

    YOLO expects BGR input; we keep the original frame for detection/tracking and
    return an RGB copy for downstream ReID feature extraction.
    """
    rgb_for_reid = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame, conf=0.05, verbose=False)
    result = results[0]

    if not result.boxes:
        return rgb_for_reid, np.empty((0, 6)), np.empty((0, 4))

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy().astype(int)

    if person_class_id is not None:
        person_mask = labels == person_class_id
        boxes = boxes[person_mask]
        confs = confs[person_mask]
        labels = labels[person_mask]

    if len(boxes) > 0:
        detections = np.column_stack((boxes, confs, labels))
    else:
        detections = np.empty((0, 6))
        boxes = np.empty((0, 4))

    return rgb_for_reid, detections, boxes


def extract_reid_features(reid_extractor, rgb, boxes, detections):
    """Extract ReID features and filter detections."""
    if len(boxes) == 0:
        return detections, boxes, None

    features, keep_idx = reid_extractor.extract_from_detections(rgb, boxes)
    if keep_idx.size != len(boxes):
        detections = detections[keep_idx]
        boxes = boxes[keep_idx]

    return detections, boxes, features


def _compute_detection_quality(
        rgb_frame: np.ndarray,
        boxes: np.ndarray,
        detections: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute quality signals for each detection used in weighted aggregation."""
    if len(boxes) == 0:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty, empty

    conf_scores = np.clip(detections[:, 4].astype(np.float32, copy=False), 0.0, 1.0)
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    frame_area = float(gray.shape[0] * gray.shape[1]) if gray.size else 1.0
    frame_area = max(frame_area, 1.0)

    sharpness_vals = np.zeros(len(boxes), dtype=np.float32)
    area_vals = np.zeros(len(boxes), dtype=np.float32)
    h, w = gray.shape[:2]

    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        x1i = int(np.clip(np.floor(x1), 0, w - 1))
        y1i = int(np.clip(np.floor(y1), 0, h - 1))
        x2i = int(np.clip(np.ceil(x2), 0, w))
        y2i = int(np.clip(np.ceil(y2), 0, h))
        if x2i <= x1i or y2i <= y1i:
            continue

        crop = gray[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            continue
        sharpness_vals[idx] = float(cv2.Laplacian(crop, cv2.CV_32F).var())
        box_area = float((x2i - x1i) * (y2i - y1i))
        area_vals[idx] = min(1.0, np.sqrt(max(box_area, 1.0) / frame_area))

    # Normalize sharpness with a saturating transform.
    sharpness_scores = sharpness_vals / (sharpness_vals + 100.0)
    quality = (0.6 * conf_scores) + (0.3 * sharpness_scores) + (0.1 * area_vals)
    quality = np.clip(quality, 0.0, 1.0).astype(np.float32, copy=False)
    return quality, conf_scores, sharpness_scores.astype(np.float32, copy=False)


def convert_to_fiftyone_detections(
        tracks,
        features,
        person_label,
        width,
        height,
        quality_scores: Optional[np.ndarray] = None,
        det_conf_scores: Optional[np.ndarray] = None,
        sharpness_scores: Optional[np.ndarray] = None,
        timestamp_sec: Optional[float] = None,
):
    """Convert tracker results to FiftyOne Detection objects."""
    frame_detections = []

    if tracks.shape[0] == 0:
        return frame_detections

    # Normalize features to unit length and keep indexable by detection order
    processed_features = []
    if features is not None:
        feats = np.asarray(features, dtype=np.float32)
        # Ensure 2D: (N, D)
        if feats.ndim == 1:
            feats = feats[None, :]
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid divide-by-zero
        feats = feats / norms
        processed_features = [f for f in feats]

    for track in tracks:
        x1, y1, x2, y2, track_id, conf, _, _ = track
        rel_box = [
            x1 / width,
            y1 / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]

        det_index = int(track[7]) if track.shape[0] >= 8 else None
        embedding = None
        quality = None
        det_conf = None
        sharpness = None
        if processed_features and det_index is not None and 0 <= det_index < len(processed_features):
            # ByteTrack returns the detection index in the last column; use it to
            # keep embeddings aligned even when the tracker reorders outputs.
            embedding = processed_features[det_index]
            if quality_scores is not None and det_index < len(quality_scores):
                quality = float(quality_scores[det_index])
            if det_conf_scores is not None and det_index < len(det_conf_scores):
                det_conf = float(det_conf_scores[det_index])
            if sharpness_scores is not None and det_index < len(sharpness_scores):
                sharpness = float(sharpness_scores[det_index])

        det_kwargs = {}
        if quality is not None:
            det_kwargs["quality"] = quality
        if det_conf is not None:
            det_kwargs["det_confidence"] = det_conf
        if sharpness is not None:
            det_kwargs["sharpness"] = sharpness
        if timestamp_sec is not None:
            det_kwargs["timestamp_sec"] = float(timestamp_sec)

        frame_detections.append(
            fo.Detection(
                label=person_label,
                bounding_box=rel_box,
                confidence=conf,
                index=int(track_id),
                embeddings=embedding,
                **det_kwargs,
            )
        )

    return frame_detections


def _extract_person_detections(result, person_class_id):
    """Build detector output arrays (Nx6 detections, Nx4 boxes) for a single frame."""
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32, copy=False)
    confs = result.boxes.conf.cpu().numpy().astype(np.float32, copy=False)
    labels = result.boxes.cls.cpu().numpy().astype(np.int32, copy=False)

    if person_class_id is not None:
        person_mask = labels == person_class_id
        boxes = boxes[person_mask]
        confs = confs[person_mask]
        labels = labels[person_mask]

    if len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

    detections = np.column_stack((boxes, confs, labels.astype(np.float32, copy=False)))
    return detections, boxes


def _process_frame_batch(
        *,
        frames: List[np.ndarray],
        frame_numbers: List[int],
        fps: float,
        model,
        person_class_id,
        person_label,
        reid_extractor,
        tracker,
        sample,
        width: int,
        height: int,
        show_visuals: bool,
) -> bool:
    """
    Process a batch of frames:
    - Batched YOLO inference for GPU efficiency
    - Sequential tracker updates to preserve tracker state semantics
    Returns True when user requests quit via visualization window.
    """
    results = model(frames, conf=0.05, verbose=False)

    for frame, frame_number, result in zip(frames, frame_numbers, results):
        detections, boxes = _extract_person_detections(result, person_class_id)
        timestamp_sec = max(float(frame_number - 1), 0.0) / max(fps, 1e-6)
        quality_scores = None
        det_conf_scores = None
        sharpness_scores = None

        if len(boxes) > 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections, boxes, features = extract_reid_features(
                reid_extractor, rgb, boxes, detections
            )
            quality_scores, det_conf_scores, sharpness_scores = _compute_detection_quality(
                rgb, boxes, detections
            )
        else:
            features = None

        tracks = tracker.update(detections, frame, features)

        if show_visuals:
            tracker.plot_results(frame, show_trajectories=True)
            cv2.imshow("BoXMOT + Ultralytics", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return True

        frame_detections = convert_to_fiftyone_detections(
            tracks,
            features,
            person_label,
            width,
            height,
            quality_scores=quality_scores,
            det_conf_scores=det_conf_scores,
            sharpness_scores=sharpness_scores,
            timestamp_sec=timestamp_sec,
        )
        sample.frames[frame_number]["detections"] = fo.Detections(
            detections=frame_detections
        )

    return False


def process_single_video(sample, model, person_class_id, person_label,
                         reid_extractor, tracker, show_visuals,
                         det_batch_size: int = 8):
    """Process a single video file and add detections to the sample."""
    cap = cv2.VideoCapture(sample.filepath)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 30.0

    if width == 0 or height == 0:
        print(f"Skipping corrupt or empty video: {sample.filepath}")
        cap.release()
        return False

    sample.frames.clear()
    frame_number = 0
    frame_buffer: List[np.ndarray] = []
    frame_numbers: List[int] = []
    det_batch_size = max(1, int(det_batch_size))
    stop_requested = False

    with torch.inference_mode():
        while True:
            success, frame = cap.read()
            if not success:
                if frame_buffer:
                    stop_requested = _process_frame_batch(
                        frames=frame_buffer,
                        frame_numbers=frame_numbers,
                        fps=fps,
                        model=model,
                        person_class_id=person_class_id,
                        person_label=person_label,
                        reid_extractor=reid_extractor,
                        tracker=tracker,
                        sample=sample,
                        width=width,
                        height=height,
                        show_visuals=show_visuals,
                    )
                break

            frame_number += 1
            frame_buffer.append(frame)
            frame_numbers.append(frame_number)

            if len(frame_buffer) >= det_batch_size:
                stop_requested = _process_frame_batch(
                    frames=frame_buffer,
                    frame_numbers=frame_numbers,
                    fps=fps,
                    model=model,
                    person_class_id=person_class_id,
                    person_label=person_label,
                    reid_extractor=reid_extractor,
                    tracker=tracker,
                    sample=sample,
                    width=width,
                    height=height,
                    show_visuals=show_visuals,
                )
                frame_buffer = []
                frame_numbers = []
                if stop_requested:
                    break

    cap.release()
    sample.save()
    print(f"Processed and saved detections for {sample.filepath}")
    return not stop_requested


def process_video_file(
        dataset,
        show_visuals: bool = False,
        det_batch_size: int = 8,
        reid_model_name: str = "osnet_ain_x1_0",
        reid_model_path: str = "",
        tracker_type: str = "bytetrack",
        tracker_reid_weights: Optional[str] = None,
        tracker_half: bool = False,
):
    """Process all videos in the dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load components
    model, person_class_id = load_detector("yolo11m.pt")
    reid_extractor = load_reid_extractor(
        model_name=reid_model_name,
        model_path=reid_model_path,
        device=device.type,
    )

    person_label = model.names.get(person_class_id, "person")

    # Process each video
    for sample in dataset.iter_samples(progress=True):
        tracker = load_tracker(
            tracker_type=tracker_type,
            device=device.type,
            tracker_reid_weights=tracker_reid_weights,
            tracker_half=tracker_half,
        )
        success = process_single_video(
            sample, model, person_class_id, person_label,
            reid_extractor, tracker, show_visuals,
            det_batch_size=det_batch_size,
        )
        if not success:
            break

    if show_visuals:
        cv2.destroyAllWindows()


def load_video_files(fo_dataset_name, dataset_dir, overwrite):
    """Load or create a FiftyOne dataset from video files."""
    fo_datasets = fo.list_datasets()
    new_dataset = True

    if fo_dataset_name in fo_datasets and not overwrite:
        dataset = fo.load_dataset(fo_dataset_name)
        new_dataset = False
    else:
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.VideoDirectory,
            name=fo_dataset_name,
            persistent=True,
            overwrite=overwrite,
        )

    return dataset, new_dataset


def configure_dataset_visualization(dataset):
    """Configure FiftyOne dataset color scheme."""
    dataset.app_config.color_scheme = fo.ColorScheme(
        color_by="value",
        fields=[
            {
                "path": "frames.detections",
                "colorByAttribute": "index",
            }
        ]
    )

def get_frame_view(dataset):
    return dataset.to_frames(sample_frames=True, output_dir='/tmp')


def compute_similarity(frame_view, sim_key):
    return fob.compute_similarity(
        frame_view,
        patches_field='detections',
        embeddings_field='embeddings',  # precomputed patch embeddings
        brain_key=sim_key,
        backend="sklearn",  # Explicitly set backend (default anyway)
        metric="cosine"     # Passed to SklearnSimilarityConfig; this is the default, but explicit for clarity
    )

def compute_visualization(frame_view, sim_key, viz_key):
    fob.compute_visualization(
        samples=frame_view,
        patches_field='detections',
        similarity_index=sim_key,
        num_dims=2,
        method="umap",
        brain_key=viz_key,
        verbose=True,
        seed=51,
        metric="cosine"  # Passed to UmapVisualizationConfig to override default 'euclidean'
    )

def launch_app(frame_view):
    patches_view = frame_view.to_patches(field='detections')
    session = fo.launch_app(patches_view)
    session.wait()


def time_stage(timings: dict, stage_name: str, fn):
    t0 = time.perf_counter()
    result = fn()
    timings[stage_name] = time.perf_counter() - t0
    return result


def write_timing_report(timings: dict, output_file: str = "timings.json"):
    if not timings:
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2)

    total = sum(timings.values())
    print("\n=== Pipeline Timings ===")
    for stage, elapsed in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        pct = (100.0 * elapsed / total) if total else 0.0
        print(f"{stage:22s}: {elapsed:8.2f}s ({pct:5.1f}%)")
    print(f"{'total':22s}: {total:8.2f}s")
    print(f"Timing report saved to: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a FiftyOne dataset from a video directory.")
    parser.add_argument("--fo-dataset-name", default="re_id", help="Name of the dataset")
    parser.add_argument("--dataset-dir", required=True, help="Path to the videos directory")
    parser.add_argument("--show", action="store_true", help="Show live video visualization during processing")
    parser.add_argument('--overwrite-loading', action='store_true', help='reload fo dataset')
    parser.add_argument('--overwrite-algo', action='store_true', help='recompute embedding')
    parser.add_argument('--save-visual-debug', action='store_true', help='Save cropped images of each person for visual debugging')
    parser.add_argument('--visual-debug-dir', default='debug_visual', help='Directory to save debug images')
    parser.add_argument('--sim-key', default='embd_sim', help='Brain key for similarity index')
    parser.add_argument('--viz-key', default='embd_viz', help='Brain key for visualization')
    parser.add_argument('--det-batch-size', type=int, default=8, help='Batch size for YOLO frame inference')
    parser.add_argument('--reid-model-name', default='osnet_ain_x1_0', help='Torchreid model name')
    parser.add_argument('--reid-model-path', default='', help='Optional fine-tuned ReID checkpoint path')
    parser.add_argument(
        '--tracker-type',
        default='bytetrack',
        choices=['bytetrack', 'strongsort', 'deepocsort'],
        help='Tracker backend',
    )
    parser.add_argument(
        '--tracker-reid-weights',
        default='',
        help='ReID weights for StrongSORT/DeepOCSORT (defaults to BoxMOT osnet_x0_25_msmt17.pt)',
    )
    parser.add_argument('--tracker-half', action='store_true', help='Use half precision for tracker ReID model (CUDA only)')
    parser.add_argument('--disable-sparse-clustering', action='store_true', help='Disable sparse neighbor clustering path')
    parser.add_argument('--sparse-threshold', type=int, default=1000, help='Tracklet count threshold to enable sparse clustering')
    parser.add_argument('--linkage', default='min', choices=['min', 'mean', 'representative'], help='Tracklet linkage for distance computation')
    parser.add_argument('--min-tracklet-frames', type=int, default=2, help='Minimum frames per tracklet')
    parser.add_argument('--disable-quality-weighting', action='store_true', help='Disable confidence/sharpness weighted representative embeddings')
    parser.add_argument('--quality-alpha', type=float, default=0.75, help='Weight of quality scores in representative aggregation [0,1]')
    parser.add_argument('--disable-temporal-smoothing', action='store_true', help='Disable temporal smoothing of embeddings within each tracklet')
    parser.add_argument('--smoothing-window', type=int, default=5, help='Temporal smoothing window size (odd recommended)')
    parser.add_argument('--temporal-penalty', type=float, default=0.05, help='Soft penalty on temporal-center mismatch across clips')
    parser.add_argument('--temporal-max-gap-sec', type=float, default=None, help='Optional hard max gap (seconds) when absolute clip timestamps are parseable')
    parser.add_argument('--motion-penalty', type=float, default=0.05, help='Soft penalty on motion-profile mismatch')
    parser.add_argument('--disable-postprocess-merge', action='store_true', help='Disable post-clustering singleton merge pass')
    parser.add_argument('--postprocess-merge-epsilon', type=float, default=None, help='Distance threshold for post-clustering merges')
    return parser.parse_args()


def main():
    seed_everything(51)
    args = parse_args()
    timings = {}

    dataset, new_dataset = time_stage(
        timings,
        "load_dataset",
        lambda: load_video_files(
            fo_dataset_name=args.fo_dataset_name,
            dataset_dir=args.dataset_dir,
            overwrite=args.overwrite_loading
        ),
    )

    if new_dataset or args.overwrite_algo:
        time_stage(
            timings,
            "video_processing",
            lambda: process_video_file(
                dataset,
                show_visuals=args.show,
                det_batch_size=args.det_batch_size,
                reid_model_name=args.reid_model_name,
                reid_model_path=args.reid_model_path,
                tracker_type=args.tracker_type,
                tracker_reid_weights=args.tracker_reid_weights or None,
                tracker_half=args.tracker_half,
            ),
        )

    # # Configure visualization
    # configure_dataset_visualization(dataset)
    #
    # # Launch app
    # session = fo.launch_app(dataset)
    # session.wait()

    # Assume `dataset` is your processed FiftyOne video dataset
    frame_view = time_stage(
        timings,
        "build_frame_view",
        lambda: get_frame_view(dataset),
    )

    sim_key = args.sim_key
    viz_key = args.viz_key

    brain_runs = dataset.list_brain_runs()

    if sim_key in brain_runs and args.overwrite_algo:
        dataset.delete_brain_run(sim_key)

    if sim_key not in brain_runs or args.overwrite_algo:
        time_stage(
            timings,
            "similarity",
            lambda: compute_similarity(frame_view, sim_key),
        )

    if viz_key in brain_runs and args.overwrite_algo:
        dataset.delete_brain_run(viz_key)

    if viz_key not in brain_runs or args.overwrite_algo:
        time_stage(
            timings,
            "visualization",
            lambda: compute_visualization(frame_view, sim_key, viz_key),
        )

    # Build / load cached detections
    all_detections = time_stage(
        timings,
        "detection_cache",
        lambda: compute_or_load_all_detections(
            frame_view=frame_view,
            dataset=dataset,
            dataset_dir=args.dataset_dir,
            overwrite_algo=args.overwrite_algo,
        ),
    )

    time_stage(
        timings,
        "clustering",
        lambda: generate_person_catalogue(
            all_detections,
            output_file="catalogue_simple.json",
            use_sparse_neighbors=not args.disable_sparse_clustering,
            sparse_if_n_ge=args.sparse_threshold,
            linkage=args.linkage,
            min_tracklet_frames=args.min_tracklet_frames,
            use_quality_weights=not args.disable_quality_weighting,
            quality_alpha=args.quality_alpha,
            smooth_embeddings=not args.disable_temporal_smoothing,
            smoothing_window=args.smoothing_window,
            temporal_penalty=args.temporal_penalty,
            temporal_max_gap_sec=args.temporal_max_gap_sec,
            motion_penalty=args.motion_penalty,
            postprocess_merge=not args.disable_postprocess_merge,
            postprocess_merge_epsilon=args.postprocess_merge_epsilon,
        ),
    )

    time_stage(
        timings,
        "scene_classification",
        lambda: classify_scenes(dataset, all_detections),
    )
    write_timing_report(timings)

#    launch_app(frame_view)


if __name__ == "__main__":
    main()
