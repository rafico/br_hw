# Testing Re-ID Improvements

## Current Status

Running improved pipeline with:
- **Clustering epsilon**: 0.2 → 0.35 (main fix)
- **Min tracklet frames**: 3 → 2
- **Tracker settings**: max_age=60, min_hits=1, det_thresh=0.2
- **YOLO confidence**: 0.1 → 0.05

## How to Compare Results

### 1. Wait for pipeline to complete (~5-10 minutes)

### 2. Compare catalogues:
```bash
python compare_results.py
```

### 3. Run distance analysis:
```bash
python analyze_reid_performance.py
```

## Expected Improvements

### Baseline (Before):
```
Total unique persons: 25
Cross-clip matches: 2 (8%)
- Person 1: clips 1, 2, 3
- Person 2: clips 2, 3
```

### Target (After):
```
Total unique persons: 12-18
Cross-clip matches: 8-12 (50-60%)
- More realistic person count
- Better cross-clip matching
```

## Key Metrics to Check

1. **Total unique persons**: Should DECREASE (less fragmentation)
2. **Cross-clip matches**: Should INCREASE significantly
3. **Cross-clip match rate**: Should go from ~8% to ~50-60%

## If Results Still Poor

Try these additional adjustments:

### Option 1: Even more relaxed epsilon
```python
cluster_selection_epsilon=0.40  # from 0.35
```

### Option 2: Use mean instead of min linkage
```python
linkage="mean"  # from "min"
```

### Option 3: Lower min_cluster_size
```python
min_cluster_size=1  # from 2 (allow all matches)
```

### Option 4: Disable min_tracklet_frames entirely
```python
min_tracklet_frames=1  # from 2
```

## Running Manual Tests

To test different epsilon values quickly:
```python
from compute_or_load_all_detections import compute_or_load_all_detections
from generate_person_catalogue import generate_person_catalogue
import fiftyone as fo

# Load dataset
dataset = fo.load_dataset("re_id_improved")
frame_view = dataset.to_frames(sample_frames=True)

# Load cached detections
all_detections = compute_or_load_all_detections(
    frame_view=frame_view,
    dataset=dataset,
    dataset_dir="/home/rafi/Downloads/blackrover_hw/videos",
    overwrite_algo=False
)

# Test different epsilon values
for eps in [0.30, 0.35, 0.40, 0.45]:
    print(f"\n=== Testing epsilon={eps} ===")
    catalogue = generate_person_catalogue(
        all_detections,
        output_file=f"catalogue_eps_{eps:.2f}.json",
        cluster_selection_epsilon=eps,
    )
```

## Validation Criteria

The improvements are successful if:
- ✅ Cross-clip match rate > 40%
- ✅ Total unique persons < 20
- ✅ At least 5-6 persons appear in multiple clips
- ✅ Main characters tracked consistently across clips

If any of these fail, continue with additional tuning options above.
