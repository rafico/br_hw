import argparse
import inspect
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import fiftyone as fo
import fiftyone.brain as fob
import numpy as np
import torch
from boxmot import BotSort, ByteTrack, DeepOcSort, StrongSort
from boxmot.utils import WEIGHTS as BOXMOT_WEIGHTS
from ultralytics import YOLO

from cluster_v2 import generate_person_catalogue_v2
import evaluate as evaluate_module
from compute_or_load_all_detections import (
    compute_or_load_all_detections,
    detections_cache_path,
    save_all_detections,
)
from finetune_reid import train as finetune_reid_train
from generate_person_catalogue import generate_person_catalogue
from classify_scenes import classify_scenes
from reid_ensemble import build_extractor
from reid_model import torso_color_hist
from utils_determinism import seed_everything
from vlm_scene import classify_scenes_vlm
from visualizers import export_to_rerun


def _yolo_inference_kwargs() -> dict:
    return {
        "conf": 0.05,
        "verbose": False,
        "imgsz": 640,
        "half": bool(torch.cuda.is_available()),
    }


def load_detector(model_path: str = "yolo26m.pt"):
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

    return build_extractor(
        name=model_name,
        device=device,
        image_size=image_size,
        batch_size=batch_size,
        model_path=model_path,
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
    if tracker_device.type == "cuda" and tracker_device.index is None:
        tracker_device = torch.device("cuda:0")
    use_half = bool(tracker_half and tracker_device.type == "cuda")
    reid_weights = Path(tracker_reid_weights) if tracker_reid_weights else _default_tracker_reid_weights()

    try:
        if tracker_type == "botsort":
            return BotSort(
                reid_weights=reid_weights,
                device=tracker_device,
                half=use_half,
                with_reid=True,
                track_high_thresh=0.45,
                track_low_thresh=0.15,
                new_track_thresh=0.55,
                match_thresh=0.8,
                proximity_thresh=0.5,
                appearance_thresh=0.25,
                frame_rate=30,
                fuse_first_associate=True,
            )
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
        "Expected one of: bytetrack, botsort, strongsort, deepocsort."
    )


def _tracker_update_kw(tracker) -> Optional[str]:
    try:
        if "embs" in inspect.signature(tracker.update).parameters:
            return "embs"
    except (TypeError, ValueError):
        pass

    try:
        source = inspect.getsource(tracker.update)
    except (OSError, TypeError):
        source = ""

    return "embs" if "embs" in source else None


def update_tracker(tracker, detections, frame, features):
    tracker_update_kw = _tracker_update_kw(tracker)
    if tracker_update_kw:
        return tracker.update(detections, frame, embs=features)
    return tracker.update(detections, frame, features)


def run_detection(model, frame, person_class_id):
    """Run YOLO detection and filter for person class.

    YOLO expects BGR input; we keep the original frame for detection/tracking and
    return an RGB copy for downstream ReID feature extraction.
    """
    rgb_for_reid = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame, **_yolo_inference_kwargs())
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
        torso_hists: Optional[np.ndarray] = None,
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
        torso_hist = None
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
            if torso_hists is not None and det_index < len(torso_hists):
                torso_hist = np.asarray(torso_hists[det_index], dtype=np.float32).tolist()

        det_kwargs = {}
        if quality is not None:
            det_kwargs["quality"] = quality
        if det_conf is not None:
            det_kwargs["det_confidence"] = det_conf
        if sharpness is not None:
            det_kwargs["sharpness"] = sharpness
        if timestamp_sec is not None:
            det_kwargs["timestamp_sec"] = float(timestamp_sec)
        if torso_hist is not None:
            det_kwargs["torso_hist"] = torso_hist
        det_kwargs["box_xyxy_abs"] = [float(x1), float(y1), float(x2), float(y2)]
        det_kwargs["frame_width"] = int(width)
        det_kwargs["frame_height"] = int(height)

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
    results = model(frames, **_yolo_inference_kwargs())

    for frame, frame_number, result in zip(frames, frame_numbers, results):
        detections, boxes = _extract_person_detections(result, person_class_id)
        timestamp_sec = max(float(frame_number - 1), 0.0) / max(fps, 1e-6)
        quality_scores = None
        det_conf_scores = None
        sharpness_scores = None
        torso_hists = None

        if len(boxes) > 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections, boxes, features = extract_reid_features(
                reid_extractor, rgb, boxes, detections
            )
            quality_scores, det_conf_scores, sharpness_scores = _compute_detection_quality(
                rgb, boxes, detections
            )
            torso_hists = np.stack(
                [torso_color_hist(rgb, box) for box in boxes],
                axis=0,
            ).astype(np.float32)
        else:
            features = None

        tracks = update_tracker(tracker, detections, frame, features)

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
            torso_hists=torso_hists,
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
        yolo_weights: str = "yolo26m.pt",
        reid_model_name: str = "osnet_ain_x1_0",
        reid_backbone: Optional[str] = None,
        reid_model_path: str = "",
        tracker_type: str = "bytetrack",
        tracker_reid_weights: Optional[str] = None,
        tracker_half: bool = False,
):
    """Process all videos in the dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load components
    model, person_class_id = load_detector(yolo_weights)
    reid_extractor = load_reid_extractor(
        model_name=reid_backbone or reid_model_name,
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


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def _load_rgb_frame_for_reembedding(video_path: str, frame_num: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(frame_num) - 1, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _merge_clip_embedding(existing_embedding, clip_embedding: np.ndarray, reid_backbone: str) -> np.ndarray:
    clip_embedding = _normalize_rows(np.asarray(clip_embedding, dtype=np.float32)[None, :])[0]
    normalized_backbone = str(reid_backbone).lower()
    if normalized_backbone == "clipreid":
        return clip_embedding.astype(np.float32, copy=False)

    if normalized_backbone != "ensemble":
        raise ValueError(
            f"Fine-tune re-embedding only supports clipreid or ensemble backbones, got {reid_backbone!r}"
        )

    existing = np.asarray(existing_embedding, dtype=np.float32).reshape(-1)
    if existing.size <= clip_embedding.size:
        return clip_embedding.astype(np.float32, copy=False)

    osnet_part = existing[:-clip_embedding.size]
    osnet_part = _normalize_rows(osnet_part[None, :])[0]
    merged = np.concatenate([osnet_part, clip_embedding], axis=0)
    return _normalize_rows(merged[None, :])[0]


def reembed_detections_with_finetuned_clip(
        all_detections,
        *,
        reid_backbone: str,
        clip_weights_path: str,
        device: Optional[str] = None,
):
    if not all_detections:
        return []

    if str(reid_backbone).lower() not in {"clipreid", "ensemble"}:
        raise ValueError("--finetune-reid requires --reid-backbone clipreid or ensemble")

    clip_extractor = load_reid_extractor(
        model_name="clipreid",
        model_path=clip_weights_path,
        device=device,
    )

    updated = [dict(det) for det in all_detections]
    grouped_indices = defaultdict(list)
    for idx, det in enumerate(all_detections):
        grouped_indices[(det["video_path"], str(det["clip_id"]), int(det["frame_num"]))].append(idx)

    with torch.inference_mode():
        for (video_path, _clip_id, frame_num), indices in sorted(grouped_indices.items()):
            frame_rgb = _load_rgb_frame_for_reembedding(video_path, frame_num)
            if frame_rgb is None:
                continue

            boxes = np.asarray([all_detections[idx]["box_xyxy_abs"] for idx in indices], dtype=np.float32)
            features, keep_idx = clip_extractor.extract_from_detections(frame_rgb, boxes)
            feature_map = {
                int(det_offset): np.asarray(features[row], dtype=np.float32)
                for row, det_offset in enumerate(np.asarray(keep_idx, dtype=np.int64))
            }

            for det_offset, det_idx in enumerate(indices):
                clip_feature = feature_map.get(int(det_offset))
                if clip_feature is None:
                    continue
                updated[det_idx]["embeddings"] = _merge_clip_embedding(
                    all_detections[det_idx]["embeddings"],
                    clip_feature,
                    reid_backbone=reid_backbone,
                )

    return updated


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Create a FiftyOne dataset from a video directory.")
    parser.add_argument("--fo-dataset-name", default="re_id", help="Name of the dataset")
    parser.add_argument("--dataset-dir", required=True, help="Path to the videos directory")
    parser.add_argument("--yolo-weights", default="yolo26m.pt", help="Detector weights to load via Ultralytics")
    parser.add_argument("--show", action="store_true", help="Show live video visualization during processing")
    parser.add_argument('--overwrite-loading', action='store_true', help='reload fo dataset')
    parser.add_argument('--overwrite-algo', action='store_true', help='recompute embedding')
    parser.add_argument('--save-visual-debug', action='store_true', help='Save cropped images of each person for visual debugging')
    parser.add_argument('--visual-debug-dir', default='debug_visual', help='Directory to save debug images')
    parser.add_argument('--sim-key', default='embd_sim', help='Brain key for similarity index')
    parser.add_argument('--viz-key', default='embd_viz', help='Brain key for visualization')
    parser.add_argument('--det-batch-size', type=int, default=8, help='Batch size for YOLO frame inference')
    parser.add_argument('--reid-model-name', default='osnet_ain_x1_0', help='Torchreid model name')
    parser.add_argument(
        '--reid-backbone',
        default='ensemble',
        choices=['osnet_ain', 'clipreid', 'ensemble'],
        help='ReID feature extractor backend',
    )
    parser.add_argument('--reid-model-path', default='', help='Optional fine-tuned ReID checkpoint path')
    parser.add_argument('--reid-weights', default='', help='Override CLIP ReID checkpoint path')
    parser.add_argument(
        '--tracker-type',
        default='botsort',
        choices=['bytetrack', 'botsort', 'strongsort', 'deepocsort'],
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
    parser.add_argument('--use-rerank', action='store_true', help='Use k-reciprocal re-ranking on representative embeddings')
    parser.add_argument('--cooccurrence-constraint', action='store_true', help='Allow same-clip merges only when tracklets do not overlap')
    parser.set_defaults(use_new_clustering=True)
    parser.add_argument('--use-new-clustering', dest='use_new_clustering', action='store_true', help='Use the HDBSCAN multi-prototype clustering pipeline')
    parser.add_argument('--legacy-clustering', dest='use_new_clustering', action='store_false', help='Use the legacy catalogue_simple clustering pipeline')
    parser.add_argument('--pose-filter', action='store_true', help='Enable MediaPipe pose filtering for V2 keyframe selection')
    parser.add_argument('--scene-backend', default='gemini', choices=['gemini', 'videomae', 'internvideo'], help='Scene classification backend')
    parser.add_argument(
        '--visualizer',
        default='fiftyone',
        choices=['none', 'fiftyone', 'rerun', 'both'],
        help='Post-run visualization backend',
    )
    parser.add_argument('--rerun-spawn', action='store_true', help='Spawn a Rerun viewer after export')
    parser.add_argument('--rerun-save', default='', help='Optional path to save a Rerun .rrd recording')
    parser.add_argument('--rerun-sample-every', type=int, default=1, help='Sample every Nth frame when exporting frames to Rerun')
    parser.add_argument('--rerun-max-frames-per-clip', type=int, default=None, help='Optional cap on exported frame images per clip')
    parser.add_argument('--finetune-reid', action='store_true', help='Run a self-bootstrap CLIP fine-tune pass before the final clustering pass')
    parser.add_argument('--finetune-min-frames', type=int, default=30, help='Minimum total frames across appearances for pseudo-label fine-tuning')
    parser.add_argument('--finetune-min-prob', type=float, default=0.9, help='Minimum cluster_probability gate for pseudo-label fine-tuning')
    parser.add_argument('--finetune-epochs', type=int, default=5, help='Epoch count for pseudo-label CLIP fine-tuning')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation against ground_truth.json if available')
    return parser.parse_args()


def build_video_meta(dataset) -> dict:
    meta = {}
    for sample in dataset.iter_samples(progress=False):
        cap = cv2.VideoCapture(sample.filepath)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps <= 1e-6:
            fps = 30.0
        meta[Path(sample.filepath).stem] = {
            "fps": fps,
            "frame_count": frame_count,
        }
    return meta


def main():
    seed_everything(51)
    args = parse_args()
    timings = {}

    if args.finetune_reid and not args.use_new_clustering:
        raise ValueError("--finetune-reid requires the stage-2 clustering path")
    if args.finetune_reid and args.reid_backbone not in {"clipreid", "ensemble"}:
        raise ValueError("--finetune-reid requires --reid-backbone clipreid or ensemble")

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
                yolo_weights=args.yolo_weights,
                reid_model_name=args.reid_model_name,
                reid_backbone=args.reid_backbone,
                reid_model_path=args.reid_weights or args.reid_model_path,
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

    if args.visualizer in {"fiftyone", "both"}:
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
            reid_backbone=args.reid_backbone,
        ),
    )

    catalogue_output = "catalogue_v2.json" if args.use_new_clustering else "catalogue_simple.json"
    scene_output = "scene_labels_v2.json" if args.use_new_clustering else "scene_labels.json"
    video_meta = None

    if args.use_new_clustering:
        video_meta = time_stage(
            timings,
            "video_meta",
            lambda: build_video_meta(dataset),
        )

    def run_clustering_stage(stage_name: str):
        if args.use_new_clustering:
            return time_stage(
                timings,
                stage_name,
                lambda: generate_person_catalogue_v2(
                    all_detections,
                    video_meta=video_meta,
                    output_file=catalogue_output,
                    use_rerank=True,
                    cooccurrence_constraint=True,
                    use_pose=args.pose_filter,
                ),
            )

        return time_stage(
            timings,
            stage_name,
            lambda: generate_person_catalogue(
                all_detections,
                output_file=catalogue_output,
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
                use_rerank=args.use_rerank,
                cooccurrence_constraint=args.cooccurrence_constraint,
            ),
        )

    run_clustering_stage("clustering_pass1" if args.finetune_reid else "clustering")

    if args.finetune_reid:
        finetuned_weights = time_stage(
            timings,
            "reid_finetune",
            lambda: finetune_reid_train(
                detections=all_detections,
                catalogue_path=catalogue_output,
                output_weights="checkpoints/clipreid_ft.pth",
                epochs=args.finetune_epochs,
                min_frames=args.finetune_min_frames,
                min_probability=args.finetune_min_prob,
            ),
        )

        if finetuned_weights is not None:
            ft_cache_path = detections_cache_path(
                args.dataset_dir,
                dataset,
                reid_backbone=args.reid_backbone,
                variant="ft",
            )

            def rebuild_finetuned_cache():
                updated = reembed_detections_with_finetuned_clip(
                    all_detections,
                    reid_backbone=args.reid_backbone,
                    clip_weights_path=str(finetuned_weights),
                )
                save_all_detections(ft_cache_path, updated)
                print(f"[cache] Saved fine-tuned detections to {ft_cache_path}")
                return updated

            all_detections = time_stage(
                timings,
                "reid_reembed",
                rebuild_finetuned_cache,
            )
            run_clustering_stage("clustering_pass2")

    if args.scene_backend == "videomae":
        time_stage(
            timings,
            "scene_classification",
            lambda: classify_scenes(
                dataset,
                all_detections,
                output_file=scene_output,
                catalogue_file=catalogue_output,
            ),
        )
    else:
        time_stage(
            timings,
            "scene_classification",
            lambda: classify_scenes_vlm(
                dataset,
                catalogue_path=catalogue_output,
                output_file=scene_output,
                backend=args.scene_backend,
            ),
        )

    if args.visualizer in {"rerun", "both"}:
        rerun_output = args.rerun_save or ("" if args.rerun_spawn else "recording.rrd")
        catalogue_payload = _read_json(catalogue_output) if Path(catalogue_output).exists() else {}
        scene_payload = _read_json(scene_output) if Path(scene_output).exists() else []
        time_stage(
            timings,
            "rerun_export",
            lambda: export_to_rerun(
                detections=all_detections,
                catalogue_payload=catalogue_payload,
                scene_payload=scene_payload,
                output_rrd=rerun_output or None,
                spawn_viewer=args.rerun_spawn,
                seed=51,
                sample_every=args.rerun_sample_every,
                max_frames_per_clip=args.rerun_max_frames_per_clip,
            ),
        )

    if args.evaluate:
        time_stage(
            timings,
            "evaluation",
            lambda: evaluate_module.run(
                gt_path="ground_truth.json",
                pred_catalogue=catalogue_output,
                pred_scene=scene_output,
            ),
        )
    write_timing_report(timings)

#    launch_app(frame_view)


if __name__ == "__main__":
    main()
