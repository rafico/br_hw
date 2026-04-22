# QA Plan

This repo uses a two-part release gate:

1. Automated QA to catch structural regressions early.
2. Manual visual QA to verify tracking, clustering, and visualization behavior on real frames.

The manual checklist lives in [qa/manual_visual_checklist.md](/home/rafi/code/br_hw/qa/manual_visual_checklist.md:1).

## Automated Gate

Run the automated suites first:

```bash
make qa-offline
make qa-dataset-smoke VIDEOS=/path/to/videos
make qa-release-local VIDEOS=/path/to/videos
make qa-validate
```

Use `make qa-release-local` for the full local release matrix. It exercises the legacy path, stage-2 path, fine-tune path, Rerun export, and JSON validation.

## Manual Review Prep

Generate the manual-review bundle before signoff:

```bash
make qa-manual-visual-prep VIDEOS=/path/to/videos
```

That target produces:

- `qa_artifacts/recording.rrd`
- `qa_artifacts/manual_visual_review.json`
- `qa_artifacts/manual_visual_review.md`

The JSON manifest resolves the review artifacts and samples frames across clips. The Markdown file is the reviewer worksheet for signoff.

## Manual Visual Gate

Run the commands from the checklist:

```bash
python run.py --dataset-dir "$VIDEOS" --show --visualizer none
python run.py --dataset-dir "$VIDEOS" --visualizer fiftyone
python run.py --dataset-dir "$VIDEOS" --visualizer rerun --rerun-save qa_artifacts/recording.rrd
```

Notes:

- `--visualizer fiftyone` launches the FiftyOne app at the end of the run.
- OpenCV live view should be checked for first-frame boxes, readable IDs, detector fallback boxes, and clean `q` exit.
- Rerun review should use the sampled frames from `qa_artifacts/manual_visual_review.json` or `qa_artifacts/manual_visual_review.md`.

## Signoff

Release signoff requires:

- Automated suites green.
- Manual checklist completed.
- Cross-artifact consistency confirmed for the sampled frames.
- Reviewer notes captured in `qa_artifacts/manual_visual_review.md` or an equivalent handoff note.
