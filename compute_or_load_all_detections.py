from pathlib import Path
import json
import hashlib
from typing import List, Dict, Any

def _cache_dir(dataset_dir: str) -> Path:
    p = Path(dataset_dir) / ".cache"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _cache_key_for_dataset(dataset) -> str:
    """
    Make the cache robust to simple changes by hashing a few cheap properties.
    Extend this if you want stricter invalidation.
    """
    meta = {
        "name": dataset.name,
        "media_type": getattr(dataset, "media_type", None),
        "num_samples": len(dataset),
        "version": "v1",  # bump if you change the structure
    }
    j = json.dumps(meta, sort_keys=True)
    return hashlib.md5(j.encode("utf-8")).hexdigest()

def detections_cache_path(dataset_dir: str, dataset) -> Path:
    key = _cache_key_for_dataset(dataset)
    return _cache_dir(dataset_dir) / f"{dataset.name}_{key}_all_detections.json"

def _to_serializable_embedding(emb):
    # Handles numpy arrays or any object with .tolist()
    return emb.tolist() if hasattr(emb, "tolist") else emb

def save_all_detections(path: Path, all_detections: List[Dict[str, Any]]) -> None:
    # ensure JSON-serializable embeddings
    serializable = []
    for d in all_detections:
        e = d.get("embeddings", None)
        serializable.append({**d, "embeddings": _to_serializable_embedding(e)})
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(serializable))
    tmp.replace(path)

def load_all_detections(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text())

def compute_all_detections(frame_view, dataset) -> List[Dict[str, Any]]:
    # cache clip_id per sample once
    sample_ids = {fs.sample_id for fs in frame_view}
    clip_by_id = {sid: Path(dataset[sid].filepath).stem for sid in sample_ids}

    return [
        {
            "clip_id": clip_by_id[fs.sample_id],
            "frame_num": fs.frame_number,
            "track_id": det.index,
            "embeddings": det.embeddings,  # converted on save
        }
        for fs in frame_view
        for det in getattr(fs.detections, "detections", [])
        if det.embeddings is not None
    ]

def compute_or_load_all_detections(*, frame_view, dataset, dataset_dir: str, overwrite_algo: bool):
    cache_path = detections_cache_path(dataset_dir, dataset)
    if cache_path.exists() and not overwrite_algo:
        print(f"[cache] Loading all_detections from {cache_path}")
        return load_all_detections(cache_path)

    print(f"[cache] Computing all_detections (overwrite={overwrite_algo})…")
    all_detections = compute_all_detections(frame_view, dataset)
    save_all_detections(cache_path, all_detections)
    print(f"[cache] Saved {len(all_detections)} detections to {cache_path}")
    return all_detections
