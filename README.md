# Video Person Re-Identification and Scene Classification

This repo contains a full two-part take-home pipeline:

- Part A: assign a stable global person ID across four clips
- Part B: label each clip as `normal` or `crime` with timestamped justification and involved global IDs

The default pipeline now uses YOLO26, BoT-SORT, an OSNet+CLIP ensemble, HDBSCAN multi-prototype clustering, and Gemini-grounded scene classification. Legacy behavior is still reachable with explicit flags such as `--legacy-clustering`, `--tracker-type bytetrack`, `--reid-backbone osnet_ain`, `--yolo-weights yolo11m.pt`, and `--scene-backend videomae`.

## Quickstart

```bash
python -m venv br_env
source br_env/bin/activate
pip install -U -r requirements.txt
export GEMINI_API_KEY=...
make reproduce
```

Optional Rerun export support:

```bash
pip install -r requirements-rerun.txt
```

Default outputs:

- `catalogue_v2.json`
- `scene_labels_v2.json`
- `eval_report.json`
- `eval_report.md`

If you need the stage-1 baseline-style path instead, run `make reproduce-stage1`.

## What's New vs. Baseline

- YOLO26 replaces YOLO11 as the default detector while keeping `--yolo-weights` as an override.
- BoT-SORT replaces ByteTrack for stronger appearance-aware tracking.
- ReID supports `osnet_ain`, `clipreid`, and `ensemble`.
- ReID can optionally self-bootstrap a CLIP fine-tune pass with `--finetune-reid`.
- Stage-1 clustering adds k-reciprocal reranking and the co-occurrence constraint.
- Stage-2 clustering adds top-K keyframes, k-medoids prototypes, HDBSCAN, and torso-color tie-breaks.
- Scene classification supports Gemini JSON grounding with a legacy VideoMAE path preserved behind `--scene-backend videomae`.
- Visualization backends can now be selected with `--visualizer {none,fiftyone,rerun,both}`.
- Evaluation and ablation are first-class outputs.

## Reproducibility

`make reproduce` runs the stage-2 pipeline on `$(VIDEOS)` from the `Makefile`.

```bash
make clean-cache
make reproduce
make ablation
```

Important runtime options:

- `--yolo-weights PATH`: override the detector weights, default `yolo26m.pt`
- `--reid-weights PATH`: load a fine-tuned CLIP checkpoint
- `--finetune-reid`: run a pass-1 pseudo-label fine-tune and a pass-2 reclustering step
- `--finetune-min-frames`, `--finetune-min-prob`, `--finetune-epochs`: control pseudo-label eligibility and training time
- `--pose-filter`: enable MediaPipe-based keyframe filtering in V2 clustering
- `--visualizer rerun --rerun-save recording.rrd`: export a standalone Rerun recording
- `--visualizer fiftyone`: build the FiftyOne review view and launch the app at the end of the run
- `--visualizer none`: skip the FiftyOne similarity/UMAP stages entirely
- `--evaluate`: run `evaluate.py` against `ground_truth.json`
- `--legacy-clustering`: keep the old `catalogue_simple.json` path

Examples:

```bash
python run.py --dataset-dir "$VIDEOS" --finetune-reid
python run.py --dataset-dir "$VIDEOS" --visualizer rerun --rerun-save recording.rrd
python run.py --dataset-dir "$VIDEOS" --visualizer none
```

## QA

The concrete QA process lives in [QA_PLAN.md](QA_PLAN.md).

Local entrypoints:

- `make qa-unit`: automated unit and regression tests
- `make qa-offline`: unit tests plus CLI/help smoke, no dataset required
- `make qa-dataset-smoke VIDEOS=/path/to/videos`: offline end-to-end smoke plus output validation
- `make qa-release-local VIDEOS=/path/to/videos`: local release candidate matrix including fine-tune and Rerun
- `make qa-manual-visual-prep VIDEOS=/path/to/videos`: generate the Rerun recording and sampled-frame review bundle for manual signoff
- `make qa-validate`: validate `catalogue_v2.json`, `scene_labels_v2.json`, and `eval_report.json`

Direct scripts:

```bash
python scripts/run_qa.py --suite offline
python scripts/run_qa.py --suite dataset-smoke --dataset-dir "$VIDEOS"
python scripts/run_qa.py --suite manual-visual-prep --dataset-dir "$VIDEOS" --rerun-save qa_artifacts/recording.rrd
python scripts/prepare_manual_visual_review.py --dataset-dir "$VIDEOS" --rerun-recording qa_artifacts/recording.rrd
python scripts/qa_validate_outputs.py --catalogue catalogue_v2.json --scene scene_labels_v2.json
```

## Outputs

- `catalogue_simple.json`: legacy catalogue path
- `catalogue_v2.json`: stage-2 catalogue with adaptive epsilon/gate
- `scene_labels.json`: legacy scene labels
- `scene_labels_v2.json`: Gemini/InternVideo-oriented scene labels with `rationale`
- `ground_truth.json`: annotation file for evaluation, seedable from predictions
- `recording.rrd`: optional Rerun recording when `--visualizer rerun` is used
- `ablation_report.md`: aggregated report across configured stages

## Notes

- `GEMINI_API_KEY` is required for the Gemini scene backend.
- Seed `ground_truth.json` from the current predictions with `python scripts/seed_ground_truth.py`, then hand-correct it before evaluation.
- `--finetune-reid` currently supports `--reid-backbone clipreid` and `ensemble`.
- The fine-tuned second-pass detections are cached separately under `$(VIDEOS)/.cache/*_all_detections_ft.json`.
- `scripts/ablation.py` expects `eval_report.json` to be produced by each run.
