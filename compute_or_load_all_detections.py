import json
import hashlib
from typing import List, Dict, Any
from pathlib import Path

def _cache_dir(dataset_dir: str) -> Path:
    p = Path(dataset_dir) / ".cache"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _cache_key_for_dataset(dataset, reid_backbone: str = "osnet_ain") -> str:
    """
    Make the cache robust to simple changes by hashing a few cheap properties.
    Extend this if you want stricter invalidation.
    """
    meta = {
        "name": dataset.name,
        "media_type": getattr(dataset, "media_type", None),
        "num_samples": len(dataset),
        "reid_backbone": str(reid_backbone),
        "version": "v4",  # bump if you change the structure
    }
    j = json.dumps(meta, sort_keys=True)
    return hashlib.md5(j.encode("utf-8")).hexdigest()

def detections_cache_path(
    dataset_dir: str,
    dataset,
    reid_backbone: str = "osnet_ain",
    variant: str = "",
) -> Path:
    key = _cache_key_for_dataset(dataset, reid_backbone=reid_backbone)
    suffix = f"_{variant}" if variant else ""
    return _cache_dir(dataset_dir) / f"{dataset.name}_{key}_all_detections{suffix}.json"

def _to_serializable_embedding(emb):
    # Handles numpy arrays or any object with .tolist()
    return emb.tolist() if hasattr(emb, "tolist") else emb


def _to_serializable_value(value):
    return value.tolist() if hasattr(value, "tolist") else value

def save_all_detections(path: Path, all_detections: List[Dict[str, Any]]) -> None:
    # ensure JSON-serializable arrays
    serializable = []
    for d in all_detections:
        serializable.append(
            {
                **d,
                "embeddings": _to_serializable_embedding(d.get("embeddings", None)),
                "torso_hist": _to_serializable_value(d.get("torso_hist", None)),
                "box_xyxy_abs": _to_serializable_value(d.get("box_xyxy_abs", None)),
            }
        )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(serializable))
    tmp.replace(path)

def load_all_detections(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text())


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_all_detections(frame_view, dataset) -> List[Dict[str, Any]]:
    all_detections: List[Dict[str, Any]] = []
    clip_by_id: Dict[Any, str] = {}

    for fs in frame_view:
        sample_id = fs.sample_id
        if sample_id not in clip_by_id:
            clip_by_id[sample_id] = Path(dataset[sample_id].filepath).stem

        clip_id = clip_by_id[sample_id]
        video_path = dataset[sample_id].filepath
        detections = getattr(fs.detections, "detections", [])
        for det in detections:
            if det.embeddings is None:
                continue

            bbox = getattr(det, "bounding_box", None)
            center_x = None
            center_y = None
            bbox_w = None
            bbox_h = None
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x, y, w, h = bbox
                x_f = _safe_float(x)
                y_f = _safe_float(y)
                w_f = _safe_float(w)
                h_f = _safe_float(h)
                bbox_w = w_f
                bbox_h = h_f
                if x_f is not None and w_f is not None:
                    center_x = x_f + (0.5 * w_f)
                if y_f is not None and h_f is not None:
                    center_y = y_f + (0.5 * h_f)

            det_conf = getattr(det, "det_confidence", None)
            if det_conf is None:
                det_conf = getattr(det, "confidence", None)

            all_detections.append(
                {
                    "clip_id": clip_id,
                    "video_path": video_path,
                    "frame_num": int(fs.frame_number),
                    "track_id": int(det.index),
                    "embeddings": det.embeddings,  # converted on save
                    "confidence": _safe_float(det_conf),
                    "quality": _safe_float(getattr(det, "quality", None)),
                    "sharpness": _safe_float(getattr(det, "sharpness", None)),
                    "timestamp_sec": _safe_float(getattr(det, "timestamp_sec", None)),
                    "torso_hist": getattr(det, "torso_hist", None),
                    "box_xyxy_abs": getattr(det, "box_xyxy_abs", None),
                    "frame_width": _safe_float(getattr(det, "frame_width", None)),
                    "frame_height": _safe_float(getattr(det, "frame_height", None)),
                    "center_x": center_x,
                    "center_y": center_y,
                    "bbox_w": bbox_w,
                    "bbox_h": bbox_h,
                }
            )

    return all_detections

def compute_or_load_all_detections(
    *,
    frame_view,
    dataset,
    dataset_dir: str,
    overwrite_algo: bool,
    reid_backbone: str = "osnet_ain",
):
    cache_path = detections_cache_path(
        dataset_dir,
        dataset,
        reid_backbone=reid_backbone,
    )
    if cache_path.exists() and not overwrite_algo:
        print(f"[cache] Loading all_detections from {cache_path}")
        return load_all_detections(cache_path)

    print(f"[cache] Computing all_detections (overwrite={overwrite_algo})…")
    all_detections = compute_all_detections(frame_view, dataset)
    save_all_detections(cache_path, all_detections)
    print(f"[cache] Saved {len(all_detections)} detections to {cache_path}")
    return all_detections
