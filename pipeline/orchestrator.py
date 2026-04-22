from __future__ import annotations

from pathlib import Path

import cv2

from classify_scenes import classify_scenes
from cluster_v2 import generate_person_catalogue_v2
from compute_or_load_all_detections import (
    compute_or_load_all_detections,
    detections_cache_path,
    save_all_detections,
)
import evaluate as evaluate_module
from finetune_reid import train as finetune_reid_train
from generate_person_catalogue import generate_person_catalogue
from utils_determinism import seed_everything
from visualizers import export_to_rerun
from vlm_scene import classify_scenes_vlm

from pipeline.dataset import (
    configure_dataset_visualization,
    compute_similarity,
    compute_visualization,
    get_frame_view,
    launch_app,
    load_video_files,
)
from pipeline.reembed import read_json, reembed_detections_with_finetuned_clip
from pipeline.timings import time_stage, write_timing_report
from pipeline.vision import process_video_file


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


def run_pipeline(args):
    seed_everything(51)
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
            overwrite=args.overwrite_loading,
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
        catalogue_payload = read_json(catalogue_output) if Path(catalogue_output).exists() else {}
        scene_payload = read_json(scene_output) if Path(scene_output).exists() else []
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

    if args.visualizer in {"fiftyone", "both"}:
        configure_dataset_visualization(dataset)
        launch_app(frame_view)
