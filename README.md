# Video Person Re-Identification and Scene Classification

This repo contains a full two-part take-home pipeline:

- Part A: assign a stable global person ID across four clips
- Part B: label each clip as `normal` or `crime` with timestamped justification and involved global IDs

The default pipeline now uses BoT-SORT, an OSNet+CLIP ensemble, HDBSCAN multi-prototype clustering, and Gemini-grounded scene classification. Legacy behavior is still reachable with explicit flags such as `--legacy-clustering`, `--tracker-type bytetrack`, `--reid-backbone osnet_ain`, and `--scene-backend videomae`.

## Quickstart

```bash
python -m venv br_env
source br_env/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=...
make reproduce
```

Default outputs:

- `catalogue_v2.json`
- `scene_labels_v2.json`
- `eval_report.json`
- `eval_report.md`

If you need the stage-1 baseline-style path instead, run `make reproduce-stage1`.

## What's New vs. Baseline

- BoT-SORT replaces ByteTrack for stronger appearance-aware tracking.
- ReID supports `osnet_ain`, `clipreid`, and `ensemble`.
- Stage-1 clustering adds k-reciprocal reranking and the co-occurrence constraint.
- Stage-2 clustering adds top-K keyframes, k-medoids prototypes, HDBSCAN, and torso-color tie-breaks.
- Scene classification supports Gemini JSON grounding with a legacy VideoMAE path preserved behind `--scene-backend videomae`.
- Evaluation and ablation are first-class outputs.

## Reproducibility

`make reproduce` runs the stage-2 pipeline on `$(VIDEOS)` from the `Makefile`.

```bash
make clean-cache
make reproduce
make ablation
```

Important runtime options:

- `--reid-weights PATH`: load a fine-tuned CLIP checkpoint
- `--pose-filter`: enable MediaPipe-based keyframe filtering in V2 clustering
- `--evaluate`: run `evaluate.py` against `ground_truth.json`
- `--legacy-clustering`: keep the old `catalogue_simple.json` path

## Outputs

- `catalogue_simple.json`: legacy catalogue path
- `catalogue_v2.json`: stage-2 catalogue with adaptive epsilon/gate
- `scene_labels.json`: legacy scene labels
- `scene_labels_v2.json`: Gemini/InternVideo-oriented scene labels with `rationale`
- `ground_truth.json`: empty annotation template for manual GT entry
- `ablation_report.md`: aggregated report across configured stages

## Notes

- `GEMINI_API_KEY` is required for the Gemini scene backend.
- `ground_truth.json` is intentionally a blank template; annotate it manually before running evaluation.
- `scripts/ablation.py` expects `eval_report.json` to be produced by each run.
