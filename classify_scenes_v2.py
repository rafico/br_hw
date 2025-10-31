import json
import math
import cv2
import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import fiftyone as fo
from dataclasses import dataclass
from tqdm import tqdm  # Import tqdm for progress bars

# --- Model & Heuristic Configuration ---

# 1. Action Recognition Model (for primary label)
# We use a VideoMAE model fine-tuned on the Kinetics-400 dataset.
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
NUM_FRAMES_SAMPLED = 16  # VideoMAE expects 16 frames

# Keywords to identify "crime" classes within the model's 400 labels
# We will search the model's config for labels containing these strings.
CRIME_KEYWORDS = {
    "fighting", "punch", "kick", "slap", "headbutting", "wrestling",
    "shooting", "robbery", "stealing", "pickpocketing", "assault", "theft"
}

# 2. Motion Heuristics (for justification)
FALLEN_ASPECT_RATIO_THRESHOLD = 1.5
STANDING_ASPECT_RATIO_THRESHOLD = 1.5
RUNNING_SPEED_THRESHOLD_PX_PER_SEC = 500.0


# --- Data Structure (same as before) ---

@dataclass
class DetectionInfo:
    """
    A dataclass to hold all relevant info for a single person detection
    after processing and clustering.
    """
    detection_id: str
    sample_id: str
    clip_id: str
    frame_num: int
    track_id: int
    bbox: List[float]
    confidence: float
    frame_width: int
    frame_height: int
    global_person_id: str = None

    @property
    def bbox_abs(self) -> List[int]:
        x_rel, y_rel, w_rel, h_rel = self.bbox
        x1 = int(x_rel * self.frame_width)
        y1 = int(y_rel * self.frame_height)
        w_abs = int(w_rel * self.frame_width)
        h_abs = int(h_rel * self.frame_height)
        return [x1, y1, x1 + w_abs, y1 + h_abs]

    @property
    def bbox_wh_abs(self) -> (int, int):
        _, _, w_rel, h_rel = self.bbox
        w_abs = int(w_rel * self.frame_width)
        h_abs = int(h_rel * self.frame_height)
        return w_abs, h_abs

    @property
    def bbox_center_abs(self) -> (int, int):
        x1, y1, x2, y2 = self.bbox_abs
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


# --- Action Recognition (Model-Based) Helpers ---

def load_action_recognition_model() -> Tuple[Any, Any, Any, set]:
    """
    Loads the VideoMAE processor and model from Hugging Face.
    Also dynamically builds the set of CRIME_CLASSES.
    """
    try:
        from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    except ImportError:
        print("Error: 'transformers' library not found.")
        print("Please run: pip install transformers")
        exit(1)

    print(f"Loading action recognition model: {MODEL_NAME}...")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
        model = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    except Exception as e:
        print(f"Error loading model '{MODEL_NAME}'.")
        print("Please check your internet connection and if the model name is correct.")
        print(f"Details: {e}")
        exit(1)

    model.eval()

    # Dynamically find all "crime" related labels from the model's config
    crime_classes = set()
    if model.config.id2label:
        for label_name in model.config.id2label.values():
            for keyword in CRIME_KEYWORDS:
                if keyword in label_name.lower():
                    crime_classes.add(label_name)

    if not crime_classes:
        print(f"Warning: No crime keywords {CRIME_KEYWORDS} found in model labels.")
        print("Using fallback list: {'street fighting', 'headbutting', 'punching'}")
        crime_classes = {"street fighting", "headbutting", "punching"}
    else:
        print(f"Model loaded. Identified {len(crime_classes)} crime-related classes:")
        print(f"  {crime_classes}")

    return processor, model, DEVICE, crime_classes


def sample_video_frames(video_path: str, num_frames: int) -> List[np.ndarray]:
    """
    Samples N frames uniformly from a video file using cv2.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    # Get frame indices to sample
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR (cv2 default) to RGB (model expects)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def classify_clip_action(
        video_path: str,
        processor: Any,
        model: Any,
        device: str
) -> Tuple[str, float]:
    """
    Runs the action recognition model on a video file.

    Returns:
        (predicted_label, confidence_score)
    """
    frames = sample_video_frames(video_path, NUM_FRAMES_SAMPLED)
    if not frames:
        return "undetermined", 0.0

    try:
        # Preprocess the frames
        inputs = processor(frames, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs_on_device)
            logits = outputs.logits

        # Get prediction
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]

        # Get confidence
        confidence = logits.softmax(-1)[0, predicted_class_idx].item()

        return predicted_label, confidence

    except Exception as e:
        print(f"Error during model inference for {video_path}: {e}")
        return "error", 0.0


# --- Motion Heuristic (Justification) Helpers ---

def format_timestamp(frame_number: int, fps: float) -> str:
    """Converts a frame number to a 'MM:SS' timestamp string."""
    if fps <= 0:
        return "00:00"
    total_seconds = int(frame_number / fps)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02}:{seconds:02}"


def find_anomalous_event_justification(
        clip_detections: List[DetectionInfo],
        fps: float
) -> Optional[str]:
    """
    Analyzes clip detections for the *first* sign of a fall or run.

    Returns:
        A justification string if an anomaly is found, otherwise None.
    """
    if fps <= 0:
        return None

    tracks: Dict[int, List[DetectionInfo]] = defaultdict(list)
    for det in clip_detections:
        tracks[det.track_id].append(det)

    for track_id in tracks:
        tracks[track_id].sort(key=lambda d: d.frame_num)

    fall_events = []
    run_events = []

    for track_id, dets_in_track in tracks.items():
        if len(dets_in_track) < 2:
            continue

        for i in range(1, len(dets_in_track)):
            prev_det = dets_in_track[i - 1]
            curr_det = dets_in_track[i]

            # --- Heuristic 1: Fall Detection ---
            prev_w, prev_h = prev_det.bbox_wh_abs
            curr_w, curr_h = curr_det.bbox_wh_abs

            if prev_w == 0 or curr_w == 0 or prev_h == 0 or curr_h == 0:
                continue

            is_standing = (prev_h / prev_w) > STANDING_ASPECT_RATIO_THRESHOLD
            is_fallen = (curr_w / curr_h) > FALLEN_ASPECT_RATIO_THRESHOLD

            if is_standing and is_fallen:
                fall_events.append((curr_det, prev_det))
                break  # Found a fall, stop checking this track

            # --- Heuristic 2: Running Detection ---
            frame_diff = curr_det.frame_num - prev_det.frame_num
            if frame_diff == 0: continue
            time_delta_sec = frame_diff / fps
            if time_delta_sec == 0: continue

            prev_center_x, prev_center_y = prev_det.bbox_center_abs
            curr_center_x, curr_center_y = curr_det.bbox_center_abs

            pixel_dist = math.sqrt(
                (curr_center_x - prev_center_x) ** 2 +
                (curr_center_y - prev_center_y) ** 2
            )
            speed_px_per_sec = pixel_dist / time_delta_sec

            if speed_px_per_sec > RUNNING_SPEED_THRESHOLD_PX_PER_SEC:
                run_events.append((curr_det, speed_px_per_sec))

    # --- Prioritize fall event for justification ---
    if fall_events:
        event, prev_event = min(fall_events, key=lambda e: e[0].frame_num)
        timestamp = format_timestamp(event.frame_num, fps)
        person_id = event.global_person_id or f"TrackID {event.track_id}"

        prev_w, prev_h = prev_event.bbox_wh_abs
        curr_w, curr_h = event.bbox_wh_abs

        return (
            f"anomalous motion detected at {timestamp} (Frame {event.frame_num}), "
            f"where {person_id} transitions from standing (H/W ratio ~{prev_h / prev_w:.2f}) "
            f"to fallen (W/H ratio ~{curr_w / curr_h:.2f}), suggesting a fall or struggle."
        )

    # --- If no fall, use running event ---
    if run_events:
        event, speed = min(run_events, key=lambda e: e[0].frame_num)
        timestamp = format_timestamp(event.frame_num, fps)
        person_id = event.global_person_id or f"TrackID {event.track_id}"

        return (
            f"anomalous motion detected at {timestamp} (Frame {event.frame_num}), "
            f"where {person_id} moves at high speed ({speed:.0f} px/sec), "
            f"suggesting running or fleeing."
        )

    # No anomalous event found
    return None


# --- Main Function ---

def classify_scenes(
        dataset: fo.Dataset,
        all_detections: List[DetectionInfo],
        output_file: str = "scene_labels.json"
):
    """
    Classifies each video clip as 'normal' or 'crime' using a hybrid
    model-and-heuristic approach.
    """

    print("Starting scene classification (model-based)...")

    # 1. Load the Action Recognition model and get "crime" classes
    try:
        processor, model, device, CRIME_CLASSES = load_action_recognition_model()
    except Exception as e:
        print(f"Failed to initialize action recognition model: {e}")
        print("Aborting classification.")
        return

    # 2. Get FPS for each video
    fps_map: Dict[str, float] = {}
    for sample in dataset:
        clip_id = sample.filepath.split('/')[-1].split('.')[0]
        if sample.metadata and sample.metadata.frame_rate:
            fps_map[clip_id] = sample.metadata.frame_rate
        else:
            # Fallback
            cap = cv2.VideoCapture(sample.filepath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_map[clip_id] = fps if fps > 0 else 30.0
            cap.release()

    # Convert all_detections to DetectionInfo objects if they are dicts
    # converted_detections = []
    # for det in all_detections:
    #     if isinstance(det, dict):
    #         det.pop('embeddings', None)  # Remove unexpected 'embeddings' key if present
    #         converted_detections.append(DetectionInfo(**det))
    #     else:
    #         converted_detections.append(det)
    # all_detections = converted_detections
    #
    # # 3. Group detections by clip_id
    # detections_by_clip: Dict[str, List[DetectionInfo]] = defaultdict(list)
    # for det in all_detections:
    #     detections_by_clip[det.clip_id].append(det)

    # 4. Analyze each clip
    results = []

    # Use tqdm for progress bar over the samples
    samples = list(dataset)  # Load samples into a list for tqdm
    for sample in tqdm(samples, desc="Classifying clips"):
        clip_id = sample.filepath.split('/')[-1].split('.')[0]
        video_path = sample.filepath

        # --- Part 1: Model-Based Classification ---
        model_label, confidence = classify_clip_action(
            video_path, processor, model, device
        )

        final_label = "crime" if model_label in CRIME_CLASSES else "normal"

        # --- Part 2: Heuristic-Based Justification ---
        clip_dets = detections_by_clip.get(clip_id)
        fps = fps_map.get(clip_id, 30.0)

        heuristic_justification = None
        if clip_dets:
            heuristic_justification = find_anomalous_event_justification(
                clip_dets, fps
            )

        # --- Part 3: Combine Results ---
        justification = (
            f"Clip classified as {final_label.upper()} (model prediction: "
            f"'{model_label}' with {confidence:.1%} confidence)."
        )

        if heuristic_justification:
            if final_label == "crime":
                justification += f" This aligns with {heuristic_justification}"
            else:
                justification += f" However, heuristics detected {heuristic_justification} which may indicate an anomaly not captured by the model."
                # Optional: Override to crime if heuristics find anomaly (uncomment if desired)
                # final_label = "crime"
                # justification += " Overriding classification to CRIME based on heuristics."
        else:
            if final_label == "crime":
                justification += (
                    " Model detected anomalous action, though no specific "
                    "fall or high-speed run was found by heuristics."
                )
            else:
                justification += (
                    " No anomalous motion (falls or high-speed runs) was "
                    "detected, consistent with the 'normal' classification."
                )

        results.append({
            "clip_id": clip_id,
            "label": final_label,
            "justification": justification
        })

    # 5. Write to output file
    # Sort results by clip_id for consistent output
    results.sort(key=lambda x: x['clip_id'])

    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nSuccessfully wrote model-based scene labels to {output_file}")
    except Exception as e:
        print(f"\nError writing scene labels to {output_file}: {e}")