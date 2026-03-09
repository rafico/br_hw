import os
import json
import cv2
import torch
import numpy as np
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
            frames_bgr = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_bgr.append(frame)
                frame_idx += 1

            if not frames_bgr:
                break

            collected = len(frames_bgr)
            start = frame_idx - collected

            # Convert this segment in one array operation (BGR -> RGB).
            frames_np = np.ascontiguousarray(np.stack(frames_bgr, axis=0)[..., ::-1])
            frames = [frm for frm in frames_np]

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
    catalogue_file: str = "catalogue_simple.json",
):
    """
    Classifies each video clip as 'normal' or 'crime' using a model-based approach.
    Maps local track IDs to global person IDs and generates human-readable justifications.
    """
    print("Starting scene classification (model-based)...")

    # Load person catalogue for mapping local track IDs to global person IDs
    local_to_global_map: Dict[str, Dict[int, int]] = {}
    try:
        with open(catalogue_file, "r", encoding="utf-8") as f:
            catalogue_data = json.load(f)
            catalogue = catalogue_data.get("catalogue", {})

            # Build reverse mapping: clip_id -> {local_track_id -> global_person_id}
            for global_id, appearances in catalogue.items():
                for app in appearances:
                    clip_id = str(app["clip_id"])
                    local_track_id = int(app["local_track_id"])
                    if clip_id not in local_to_global_map:
                        local_to_global_map[clip_id] = {}
                    local_to_global_map[clip_id][local_track_id] = int(global_id)

            print(f"Loaded person catalogue with {len(catalogue)} unique persons")
    except FileNotFoundError:
        print(f"Warning: Catalogue file '{catalogue_file}' not found. Using local track IDs only.")
    except Exception as e:
        print(f"Warning: Error loading catalogue: {e}. Using local track IDs only.")

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
        # Get human-readable clip name from filename
        video_path = sample.filepath
        clip_name = os.path.splitext(os.path.basename(video_path))[0]

        # Get video FPS for timestamp calculation
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        final_label, confidence, crime_segments = classify_clip_action(
            video_path, processor, model, device, crime_classes_lower, num_frames
        )

        justification = ""
        max_frames = len(sample.frames)

        if final_label == "crime":
            frame_to_people: Dict[int, Set[int]] = {}
            for frame_num in range(1, max_frames + 1):
                frame_doc = sample.frames[frame_num]
                if "detections" not in frame_doc:
                    continue
                frame_dets = frame_doc["detections"].detections
                people = {det.index for det in frame_dets if det.index is not None}
                if people:
                    frame_to_people[frame_num] = people

            justification_parts = []
            for seg in crime_segments:
                start_frame = max(1, seg["start"] + 1)
                end_frame = min(max_frames, seg["end"] + 1)

                # Calculate timestamps
                start_time = start_frame / fps
                end_time = end_frame / fps

                if end_frame < start_frame:
                    frames_involved = []
                    involved_local: Set[int] = set()
                else:
                    frames_involved = list(range(start_frame, end_frame + 1))
                    involved_local = set()
                    for frame_num in frames_involved:
                        involved_local |= frame_to_people.get(frame_num, set())

                # Map local track IDs to global person IDs
                involved_global = []
                clip_map = local_to_global_map.get(clip_name, {})
                for local_id in sorted(involved_local):
                    global_id = clip_map.get(local_id, None)
                    if global_id is not None:
                        involved_global.append(global_id)
                    else:
                        # Fallback: include local ID with note
                        involved_global.append(f"local_{local_id}")

                seg["frames"] = frames_involved
                seg["involved_people_local"] = sorted(involved_local)
                seg["involved_people_global"] = involved_global
                seg["timestamp_start"] = round(start_time, 2)
                seg["timestamp_end"] = round(end_time, 2)

                # Build human-readable justification
                people_str = ", ".join([f"person {pid}" for pid in involved_global]) if involved_global else "people"
                justification_parts.append(
                    f"{seg['label']} detected at {seg['timestamp_start']:.1f}s-{seg['timestamp_end']:.1f}s "
                    f"(frames {start_frame}-{end_frame}) involving {people_str}"
                )

                print(
                    f"Crime in {clip_name}: {seg['label']} at frames {start_frame}-{end_frame}, "
                    f"global IDs: {involved_global}"
                )

            justification = "; ".join(justification_parts)
        else:
            justification = "No criminal activity detected in the video clip."

        results.append(
            {
                "clip_id": clip_name,
                "label": final_label,
                "justification": justification,
                "max_confidence": round(confidence, 4),
                "crime_segments": crime_segments if final_label == "crime" else [],
            }
        )

    # Sort for consistent output
    results.sort(key=lambda x: x["clip_id"])

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully wrote scene labels to {output_file}")
    except Exception as e:
        print(f"\nError writing scene labels to {output_file}: {e}")
