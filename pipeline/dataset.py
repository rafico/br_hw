from __future__ import annotations

import fiftyone as fo
import fiftyone.brain as fob


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
    return dataset.to_frames(sample_frames=True, output_dir="/tmp")


def compute_similarity(frame_view, sim_key):
    return fob.compute_similarity(
        frame_view,
        patches_field="detections",
        embeddings_field="embeddings",
        brain_key=sim_key,
        backend="sklearn",
        metric="cosine",
    )


def compute_visualization(frame_view, sim_key, viz_key):
    fob.compute_visualization(
        samples=frame_view,
        patches_field="detections",
        similarity_index=sim_key,
        num_dims=2,
        method="umap",
        brain_key=viz_key,
        verbose=True,
        seed=51,
        metric="cosine",
    )


def launch_app(frame_view):
    patches_view = frame_view.to_patches(field="detections")
    session = fo.launch_app(patches_view)
    session.wait()
