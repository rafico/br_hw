import torch
from pathlib import Path
from boxmot import BotSort, BoostTrack, ByteTrack, StrongSort
import cv2
from ultralytics import YOLO
import fiftyone as fo
import fiftyone.brain as fob
import argparse
import numpy as np

from reid_model import DetectionReIDExtractor
from generate_person_catalogue import generate_person_catalogue
from compute_or_load_all_detections import compute_or_load_all_detections
from classify_scenes_v2 import classify_scenes


def load_detector(model_path: str = "yolo11m.pt"):
    """Return (ultralytics YOLO model, person_class_id)."""
    model = YOLO(model_path)
    person_class_id = next((k for k, v in model.names.items() if v == "person"), None)
    return model, person_class_id


def load_reid_extractor(
        model_name: str = "osnet_ain_x1_0",
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
        image_size=image_size,
        device=device,
        batch_size=batch_size,
        input_is_bgr=input_is_bgr,
    )


def load_tracker(device: torch.device):
    return ByteTrack(reid_weights=Path(''), device=device, half=False)


def run_detection(model, frame, person_class_id):
    """Run YOLO detection and filter for person class."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb, conf=0.1, verbose=False)
    result = results[0]

    if not result.boxes:
        return rgb, np.empty((0, 6)), np.empty((0, 4))

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

    return rgb, detections, boxes


def extract_reid_features(reid_extractor, rgb, boxes, detections):
    """Extract ReID features and filter detections."""
    if len(boxes) == 0:
        return detections, None

    features, keep_idx = reid_extractor.extract_from_detections(rgb, boxes)
    if keep_idx.size != len(boxes):
        detections = detections[keep_idx]

    return detections, features


def convert_to_fiftyone_detections(tracks, features, person_label, width, height):
    """Convert tracker results to FiftyOne Detection objects."""
    frame_detections = []

    if tracks.shape[0] == 0:
        return frame_detections

    # Normalize features to unit length
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

    for i, track in enumerate(tracks):
        x1, y1, x2, y2, track_id, conf, _, _ = track
        rel_box = [
            x1 / width,
            y1 / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]

        embedding = processed_features[i] if processed_features else None

        frame_detections.append(
            fo.Detection(
                label=person_label,
                bounding_box=rel_box,
                confidence=conf,
                index=int(track_id),
                embeddings=embedding,
            )
        )

    return frame_detections


def process_single_video(sample, model, person_class_id, person_label,
                         reid_extractor, tracker, show_visuals):
    """Process a single video file and add detections to the sample."""
    cap = cv2.VideoCapture(sample.filepath)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        print(f"Skipping corrupt or empty video: {sample.filepath}")
        cap.release()
        return False

    frame_number = 0
    sample.frames.clear()

    with torch.inference_mode():
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1

            # Run detection
            rgb, detections, boxes = run_detection(model, frame, person_class_id)

            # Extract ReID features
            detections, features = extract_reid_features(
                reid_extractor, rgb, boxes, detections
            )

            # Update tracker
            tracks = tracker.update(detections, frame)

            # Visualize if requested
            if show_visuals:
                tracker.plot_results(frame, show_trajectories=True)
                cv2.imshow("BoXMOT + Ultralytics", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Convert to FiftyOne format
            frame_detections = convert_to_fiftyone_detections(
                tracks, features, person_label, width, height
            )

            sample.frames[frame_number]["detections"] = fo.Detections(
                detections=frame_detections
            )

    cap.release()
    sample.save()
    print(f"Processed and saved detections for {sample.filepath}")
    return True


def process_video_file(dataset, show_visuals: bool = False):
    """Process all videos in the dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load components
    model, person_class_id = load_detector("yolo11m.pt")
    reid_extractor = load_reid_extractor(device=device.type)
    tracker = load_tracker(device)

    person_label = model.names.get(person_class_id, "person")

    # Process each video
    for sample in dataset.iter_samples(progress=True):
        success = process_single_video(
            sample, model, person_class_id, person_label,
            reid_extractor, tracker, show_visuals
        )

        if show_visuals and cv2.waitKey(1) & 0xFF == ord("q"):
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
    return parser.parse_args()


def main():
    args = parse_args()

    dataset, new_dataset = load_video_files(
        fo_dataset_name=args.fo_dataset_name,
        dataset_dir=args.dataset_dir,
        overwrite=args.overwrite_loading
    )

    if new_dataset or args.overwrite_algo:
        process_video_file(dataset, show_visuals=args.show)

    # # Configure visualization
    # configure_dataset_visualization(dataset)
    #
    # # Launch app
    # session = fo.launch_app(dataset)
    # session.wait()

    # Assume `dataset` is your processed FiftyOne video dataset
    frame_view = get_frame_view(dataset)

    sim_key = args.sim_key
    viz_key = args.viz_key

    brain_runs = dataset.list_brain_runs()

    if sim_key in brain_runs and args.overwrite_algo:
        dataset.delete_brain_run(sim_key)

    if sim_key not in brain_runs or args.overwrite_algo:
        sim_index = compute_similarity(frame_view, sim_key)

    if viz_key in brain_runs and args.overwrite_algo:
        dataset.delete_brain_run(viz_key)

    if viz_key not in brain_runs or args.overwrite_algo:
        compute_visualization(frame_view, sim_key, viz_key)

    # cache clip_id per sample once
    sample_ids = {fs.sample_id for fs in frame_view}
    clip_by_id = {sid: Path(dataset[sid].filepath).stem for sid in sample_ids}

    # Build / load cached detections
    all_detections = compute_or_load_all_detections(
        frame_view=frame_view,
        dataset=dataset,
        dataset_dir=args.dataset_dir,
        overwrite_algo=args.overwrite_algo,
    )

    generate_person_catalogue(
            all_detections,
            output_file="catalogue_simple.json",
    )

    classify_scenes(dataset, all_detections)

#    launch_app(frame_view)


if __name__ == "__main__":
    main()