# Video Person Re-Identification and Scene Classification

A toy system for detecting, tracking, and re-identifying persons across multiple video clips, with automatic scene classification for normal vs. crime detection.

## Features

- **Person Detection & Tracking**: Uses YOLO11 for detection and ByteTrack for multi-object tracking
- **Re-Identification (ReID)**: Extracts appearance features using OSNet to match persons across videos
- **Cross-Clip Person Catalogue**: Generates a unified catalog of unique individuals across all video clips
- **Scene Classification**: Classifies video clips as "normal" or "crime" using VideoMAE action recognition
- **Visual Analytics**: Integration with FiftyOne for interactive visualization and exploration

## Architecture

### Pipeline Overview

1. **Detection & Tracking** (`run.py`, `reid_model.py`)
   - Detects persons using YOLO11
   - Tracks them across frames with ByteTrack
   - Extracts ReID embeddings using OSNet

2. **Person Catalogue Generation** (`generate_person_catalogue.py`)
   - Groups detections into tracklets (per-clip tracks)
   - Computes representative embeddings for each tracklet
   - Clusters tracklets across clips with cross-clip constraints
   - Assigns global person IDs

3. **Scene Classification** (`classify_scenes.py`)
   - Runs VideoMAE action recognition model
   - Classifies clips based on detected actions

## Installation

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

Process videos in a directory:

```bash
python run.py --dataset-dir /path/to/videos
```

### Common Options

```bash
# Show live visualization during processing
python run.py --dataset-dir ./videos --show

# Force reprocessing
python run.py --dataset-dir ./videos --overwrite-loading --overwrite-algo
```

### Command-Line Arguments

- `--dataset-dir`: Path to directory containing video files (required)
- `--fo-dataset-name`: Name for FiftyOne dataset (default: "re_id")
- `--show`: Display live video processing visualization
- `--overwrite-loading`: Reload FiftyOne dataset from scratch
- `--overwrite-algo`: Recompute embeddings and brain runs
- `--sim-key`: Brain key for similarity index (default: "embd_sim")
- `--viz-key`: Brain key for visualization (default: "embd_viz")

## Output Files

### 1. Person Catalogue (`catalogue_simple.json`)

Contains unique persons identified across all videos:

```json
{
  "summary": {
    "total_unique_persons": 15,
    "total_tracklets": 42,
    "parameters": {...}
  },
  "catalogue": {
    "1": [
      {
        "clip_id": "video1",
        "local_track_id": 3,
        "frame_ranges": [[10, 150]],
        "num_frames": 141
      }
    ]
  }
}
```

### 2. Scene Labels (`scene_labels.json`)

Classification of each video clip:

```json
[
  {
    "clip_id": "video1",
    "label": "normal"
  },
  {
    "clip_id": "video2",
    "label": "crime"
  }
]
```

## Configuration

### ReID Model Settings

Edit `run.py` to configure the ReID extractor:

```python
reid_extractor = load_reid_extractor(
    model_name="osnet_ain_x1_0",  # Options: osnet_x1_0, osnet_ain_x1_0
    image_size=(256, 128),
    batch_size=32,
    device="cuda"
)
```

### Clustering Parameters

Adjust in `run.py` when calling `generate_person_catalogue()`:

```python
generate_person_catalogue(
    all_detections,
    min_cluster_size=2,              # Minimum tracklets per cluster
    cluster_selection_epsilon=0.025,  # Distance threshold for merging
    use_median=True                   # Use median vs. mean for representatives
)
```

### Scene Classification

Modify crime keywords in `classify_scenes.py`:

```python
CRIME_KEYWORDS = {
    "fighting", "punch", "kick", "slap", "headbutting", 
    "wrestling", "shooting", "robbery", "stealing", 
    "pickpocketing", "assault", "theft"
}
```

## Key Components

### DetectionReIDExtractor (`reid_model.py`)

Extracts appearance features from person crops:
- Uses torchreid's OSNet models
- Handles batch processing for efficiency
- Supports both BGR and RGB inputs

### Person Catalogue Generator (`generate_person_catalogue.py`)

Cross-clip person matching with constraints:
- **Co-occurrence constraint**: Persons appearing in same frame can't be matched
- **Cross-clip enforcement**: Only matches tracklets from different videos
- Custom greedy clustering algorithm with distance threshold

### Scene Classifier (`classify_scenes.py`)

Action-based scene understanding:
- Uses VideoMAE pre-trained on Kinetics-400
- Samples 16 frames uniformly from each clip
- Maps detected actions to crime/normal categories

## Advanced Features

### Caching System

Detection embeddings are cached to speed up reruns:
- Cache location: `<dataset_dir>/.cache/`
- Automatically invalidated when dataset changes
- Force recomputation with `--overwrite-algo`

### FiftyOne Integration

Visualize results interactively:

```python
import fiftyone as fo

# Load processed dataset
dataset = fo.load_dataset("re_id")

# Launch browser interface
session = fo.launch_app(dataset)
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch sizes:
```python
# In run.py
reid_extractor = load_reid_extractor(batch_size=16)  # Default: 32
```

### Model Download Issues

Models are downloaded automatically from:
- YOLO: Ultralytics Hub
- OSNet: torchreid pretrained models
- VideoMAE: Hugging Face Hub

Ensure internet connectivity on first run.

### No Crime Classes Found

The system falls back to default classes. To customize:
1. Check available Kinetics-400 labels
2. Update `CRIME_KEYWORDS` in `classify_scenes.py`



