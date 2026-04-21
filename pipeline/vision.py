from __future__ import annotations

import inspect
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import fiftyone as fo
import numpy as np
import torch
from boxmot import BotSort, ByteTrack, DeepOcSort, StrongSort
from boxmot.utils import WEIGHTS as BOXMOT_WEIGHTS
from ultralytics import YOLO

from reid_ensemble import build_extractor
from reid_model import torso_color_hist


def _yolo_inference_kwargs() -> dict:
    return {
        "conf": 0.05,
        "verbose": False,
        "imgsz": 640,
        "half": bool(torch.cuda.is_available()),
    }


def load_detector(model_path: str = "yolo26m.pt", *, yolo_cls=YOLO):
    """Return (ultralytics YOLO model, person_class_id)."""
    model = yolo_cls(model_path)
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
        *,
        bytetrack_cls=ByteTrack,
        botsort_cls=BotSort,
        strongsort_cls=StrongSort,
        deepocsort_cls=DeepOcSort,
):
    """Initialize a fresh tracker for each video to avoid cross-clip bleed-through."""
    tracker_type = tracker_type.lower()
    tracker_common_args = {
        "max_age": 60,
        "min_hits": 1,
        "det_thresh": 0.2,
    }

    if tracker_type == "bytetrack":
        return bytetrack_cls(**tracker_common_args)

    tracker_device = torch.device(device)
    if tracker_device.type == "cuda" and tracker_device.index is None:
        tracker_device = torch.device("cuda:0")
    use_half = bool(tracker_half and tracker_device.type == "cuda")
    reid_weights = Path(tracker_reid_weights) if tracker_reid_weights else _default_tracker_reid_weights()

    try:
        if tracker_type == "botsort":
            return botsort_cls(
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
            return strongsort_cls(
                reid_weights=reid_weights,
                device=tracker_device,
                half=use_half,
                **tracker_common_args,
            )
        if tracker_type == "deepocsort":
            return deepocsort_cls(
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
        return bytetrack_cls(**tracker_common_args)

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
    """Run YOLO detection and filter for person class."""
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

    processed_features = []
    if features is not None:
        feats = np.asarray(features, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats[None, :]
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
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
    """Process a batch of frames and update sample frames."""
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


def process_single_video(
        sample,
        model,
        person_class_id,
        person_label,
        reid_extractor,
        tracker,
        show_visuals,
        det_batch_size: int = 8,
):
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
        *,
        detector_loader=load_detector,
        extractor_loader=load_reid_extractor,
        tracker_loader=load_tracker,
        single_video_processor=process_single_video,
):
    """Process all videos in the dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, person_class_id = detector_loader(yolo_weights)
    reid_extractor = extractor_loader(
        model_name=reid_backbone or reid_model_name,
        model_path=reid_model_path,
        device=device.type,
    )

    person_label = model.names.get(person_class_id, "person")

    for sample in dataset.iter_samples(progress=True):
        tracker = tracker_loader(
            tracker_type=tracker_type,
            device=device.type,
            tracker_reid_weights=tracker_reid_weights,
            tracker_half=tracker_half,
        )
        success = single_video_processor(
            sample, model, person_class_id, person_label,
            reid_extractor, tracker, show_visuals,
            det_batch_size=det_batch_size,
        )
        if not success:
            break

    if show_visuals:
        cv2.destroyAllWindows()
