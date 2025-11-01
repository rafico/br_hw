import json
import cv2
import torch
import numpy as np
import fiftyone as fo
from dataclasses import dataclass
from tqdm import tqdm  # Import tqdm for progress bars
from typing import Any, Dict, List, Tuple

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


# --- Main Function ---

def classify_scenes(
        dataset: fo.Dataset,
        all_detections: List[dataclass],  # Unused now, but kept for signature
        output_file: str = "scene_labels.json"
):
    """
    Classifies each video clip as 'normal' or 'crime' using a model-based approach.
    """

    print("Starting scene classification (model-based)...")

    # 1. Load the Action Recognition model and get "crime" classes
    try:
        processor, model, device, CRIME_CLASSES = load_action_recognition_model()
    except Exception as e:
        print(f"Failed to initialize action recognition model: {e}")
        print("Aborting classification.")
        return

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

        print(f'{model_label}')

        results.append({
            "clip_id": clip_id,
            "label": final_label
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