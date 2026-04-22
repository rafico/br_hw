# Manual Visual Checklist

Run this checklist before release signoff on any change that touches tracking, clustering, or visualization.

Preparation:

1. Run `make qa-manual-visual-prep VIDEOS="$VIDEOS"`.
2. Use `qa_artifacts/manual_visual_review.json` or `qa_artifacts/manual_visual_review.md` as the source of sampled frames for the FiftyOne, Rerun, and cross-artifact checks.

## OpenCV Live View

1. Run `python run.py --dataset-dir "$VIDEOS" --show --visualizer none`.
2. Confirm boxes appear on the first visible detections, not only after track warm-up.
3. Confirm track IDs stay readable and move with the correct person.
4. Confirm detector fallback boxes appear when a person is visible but the track is not yet mature.
5. Confirm the window exits cleanly with `q`.

## FiftyOne

1. Run `python run.py --dataset-dir "$VIDEOS" --visualizer fiftyone`.
2. Sample at least 10 frames across multiple clips.
3. Confirm stored detections align with the visible person crops.
4. Confirm the same frame shows the same IDs and boxes as the cached detections.

## Rerun

1. Run `python run.py --dataset-dir "$VIDEOS" --visualizer rerun --rerun-save recording.rrd`.
2. Open the recording and inspect at least 10 sampled frames.
3. Confirm frame time, detection boxes, labels, and colors are aligned.
4. Confirm post-clustering labels use global IDs where available.
5. Confirm scene summaries reference the expected clip and timestamps.

## Cross-Artifact Consistency

1. Pick 5 frame numbers from the OpenCV or FiftyOne review.
2. Cross-check the same frames in:
   - `catalogue_v2.json`
   - `scene_labels_v2.json`
   - Rerun recording
3. Confirm clip IDs, frame numbers, track IDs, and global IDs are consistent.
