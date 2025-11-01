import os
import json
import cv2
import torch
import fiftyone as fo
from typing import Any, List, Tuple, Set, Dict
from tqdm import tqdm

# --- Model & Heuristic Configuration ---

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
CRIME_KEYWORDS = {
    "fighting", "punch", "kick", "slap", "headbutting", "wrestling",
    "shooting", "robbery", "stealing", "pickpocketing", "assault", "theft"
}

# --- Action Recognition (Model-Based) Helpers ---

def load_action_recognition_model() -> Tuple[Any, Any, str, Set[str], int]:
    """
    Loads the VideoMAE processor and model from Hugging Face.
    Also dynamically builds the set of crime-related classes (lower-cased).
    Returns: (processor, model, device, crime_classes_lower, num_frames)
    """
    try:
        from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: transformers. Install with `pip install transformers`"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
        model = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME).to(device)
    except Exception as e:
        raise RuntimeError(
            f"Error loading model '{MODEL_NAME}'. Check connectivity and model name."
        ) from e

    model.eval()

    # Discover crime-like labels from the model's label space
    crime_classes_lower: Set[str] = set()
    if getattr(model.config, "id2label", None):
        for label_name in model.config.id2label.values():
            lname = label_name.lower()
            if any(kw in lname for kw in CRIME_KEYWORDS):
                crime_classes_lower.add(lname)

    if not crime_classes_lower:
        # Fallback if none were found
        print(
            f"Warning: No crime keywords {CRIME_KEYWORDS} found in model labels. "
            "Using fallback list."
        )
        crime_classes_lower = {"street fighting", "headbutting", "punching"}

    num_frames = getattr(model.config, "num_frames", 16)

    return processor, model, device, crime_classes_lower, num_frames


def classify_clip_action(
    video_path: str,
    processor: Any,
    model: Any,
    device: str,
    crime_classes_lower: Set[str],
    num_frames: int,
) -> Tuple[str, float, List[Dict[str, Any]]]:
    """
    Runs the action recognition model on the entire video by processing
    non-overlapping segments of `num_frames` frames.

    Returns:
        (final_label, max_confidence, crime_segments)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return "undetermined", 0.0, []

    crime_segments: List[Dict[str, Any]] = []
    max_conf = 0.0
    frame_idx = 0

    try:
        while True:
            frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR -> RGB for the processor
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_idx += 1

            if not frames:
                break

            collected = len(frames)
            start = frame_idx - collected

            # Pad last frame if we ran out near EOF
            while len(frames) < num_frames:
                frames.append(frames[-1])

            try:
                inputs = processor(frames, return_tensors="pt")
                with torch.inference_mode():
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    logits = model(**inputs).logits

                pred_idx = int(logits.argmax(-1).item())
                predicted_label = model.config.id2label[pred_idx]
                predicted_label_l = predicted_label.lower()
                confidence = float(torch.softmax(logits, dim=-1)[0, pred_idx].item())

                # Decide crime: primary = discovered classes; secondary = keyword substring
                is_crime = (
                    predicted_label_l in crime_classes_lower
                    or any(kw in predicted_label_l for kw in CRIME_KEYWORDS)
                )

                if is_crime:
                    segment = {
                        "start": start,
                        "end": frame_idx - 1,
                        "label": predicted_label,
                        "conf": confidence,
                    }
                    crime_segments.append(segment)
                    max_conf = max(max_conf, confidence)

            except Exception as e:
                print(
                    f"Error during model inference for segment starting at frame {start} "
                    f"in {video_path}: {e}"
                )
    finally:
        cap.release()

    final_label = "crime" if crime_segments else "normal"
    return final_label, max_conf, crime_segments


# --- Main Function ---

def classify_scenes(
    dataset: fo.Dataset,
    all_detections: List[Any] = None,  # kept for signature; not used
    output_file: str = "scene_labels.json",
):
    """
    Classifies each video clip as 'normal' or 'crime' using a model-based approach.
    For crime segments, extracts the frames and unique people IDs (track IDs) involved in those frames.
    """
    print("Starting scene classification (model-based)...")

    try:
        processor, model, device, crime_classes_lower, num_frames = (
            load_action_recognition_model()
        )
    except Exception as e:
        print(f"Failed to initialize action recognition model: {e}")
        print("Aborting classification.")
        return

    results = []
    total = len(dataset)
    for sample in tqdm(dataset, total=total, desc="Classifying clips"):
        clip_id = str(sample.id)  # robust unique ID
        video_path = sample.filepath

        final_label, confidence, crime_segments = classify_clip_action(
            video_path, processor, model, device, crime_classes_lower, num_frames
        )

        max_frames = len(sample.frames)
        if final_label == "crime":
            for seg in crime_segments:
                involved = set()
                start_frame = seg["start"] + 1  # Convert to 1-indexed
                end_frame = seg["end"] + 1      # Convert to 1-indexed
                frames_involved = []
                for f in range(start_frame, end_frame + 1):
                    if f > max_frames:
                        break  # Safety check if segment exceeds video length
                    frames_involved.append(f)
                    if "detections" in sample.frames[f]:
                        frame_dets = sample.frames[f]["detections"].detections
                        for det in frame_dets:
                            if det.index is not None:
                                involved.add(det.index)

                seg["frames"] = frames_involved
                seg["involved_people"] = sorted(list(involved))

                print(
                    f"Crime detected in clip {clip_id}: action '{seg['label']}' "
                    f"(confidence: {seg['conf']:.2f}), frames {seg['start'] + 1}–{seg['end'] + 1}, "
                    f"involved people: {seg['involved_people']}"
                )

        results.append(
            {
                "clip_id": clip_id,
                "label": final_label,
                "max_confidence": round(confidence, 4),
                "crime_segments": crime_segments,
            }
        )

    # Sort for consistent output (by unique id string)
    results.sort(key=lambda x: x["clip_id"])

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully wrote model-based scene labels to {output_file}")
    except Exception as e:
        print(f"\nError writing scene labels to {output_file}: {e}")