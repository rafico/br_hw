import json, math, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch

# Prefer fast, reliable decoding
try:
    import decord  # type: ignore
    decord.bridge.set_bridge("torch")
    HAS_DECORD = True
except Exception:
    HAS_DECORD = False

try:
    import cv2  # fallback decoder
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

from transformers import (
    AutoModelForVideoClassification,
    AutoImageProcessor,  # newer: AutoProcessor; ImageProcessor keeps wider compatibility
)

# ---------------------
# Config / dataclasses
# ---------------------
@dataclass
class WindowPred:
    start_s: float
    end_s: float
    label: str
    score: float
    raw: Dict[str, float]  # label->prob

HF_DEFAULT_BINARY = "Nikeytas/videomae-crime-detector-ultra-v1"     # Violent vs Non-Violent (1/0)  :contentReference[oaicite:3]{index=3}
HF_DEFAULT_MULTICLASS = "OPear/videomae-large-finetuned-UCF-Crime"  # UCF-Crime labels             :contentReference[oaicite:4]{index=4}

VIOLENCE_KEYS = {
    # Keys we’ll consider “crime” for multi-class models (case-insensitive)
    "abuse", "arrest", "arson", "assault", "burglary", "explosion",
    "fighting", "robbery", "shooting", "shoplifting", "stealing", "vandalism",
}
# Some UCF-Crime models include “Road Accidents”. Treating it as crime is subjective.
# Default: NOT crime for binary label since assignment asks {normal, crime}.
TREAT_ACCIDENTS_AS_CRIME = False

# ---------------------
# Utilities
# ---------------------
def _format_ts(t: float) -> str:
    m = int(t // 60)
    s = t - m * 60
    return f"{m:02d}:{s:04.1f}"

def _safe_fps(sample) -> float:
    try:
        fps = float(getattr(getattr(sample, "metadata", None), "frame_rate", 0.0) or 0.0)
    except Exception:
        fps = 0.0
    return fps if fps > 0 else 30.0

def _video_len_and_fps(filepath: str) -> Tuple[int, float]:
    """Return (num_frames, fps) using decord or cv2."""
    if HAS_DECORD:
        vr = decord.VideoReader(filepath)
        fps = float(vr.get_avg_fps()) or 0.0
        return int(len(vr)), (fps if fps > 0 else 30.0)
    if HAS_CV2:
        cap = cv2.VideoCapture(filepath)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        return nf, (fps if fps > 0 else 30.0)
    raise RuntimeError("No video backend available. Install decord or opencv-python.")

def _sample_indices(nf: int, fps: float, start_s: float, end_s: float, num_frames: int) -> np.ndarray:
    start_f = max(0, int(round(start_s * fps)))
    end_f = max(0, int(round(end_s * fps)) - 1)
    end_f = min(end_f, nf - 1)
    if end_f < start_f:
        end_f = start_f
    if num_frames <= 1 or end_f == start_f:
        return np.array([start_f], dtype=int)
    return np.linspace(start_f, end_f, num_frames, dtype=int)

def _read_frames(filepath: str, indices: np.ndarray) -> List[np.ndarray]:
    """Returns list of RGB HxWx3 uint8 frames."""
    if HAS_DECORD:
        vr = decord.VideoReader(filepath)
        idx = [int(i) for i in indices]
        batch = vr.get_batch(idx).asnumpy()  # (T,H,W,C), uint8, RGB
        return [batch[i] for i in range(batch.shape[0])]
    # CV2 fallback
    if not HAS_CV2:
        raise RuntimeError("Install decord or opencv-python to read videos.")
    cap = cv2.VideoCapture(filepath)
    frames = []
    last_pos = -1
    for fi in map(int, indices):
        if fi != last_pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        last_pos = fi
    cap.release()
    return frames

def _load_global_map(path: str | Path = "catalogue_simple.json") -> Dict[Tuple[str, int], str]:
    """(clip_id, local_track_id) -> global_id; robust to simple schema variations."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    m: Dict[Tuple[str, int], str] = {}
    if isinstance(data, dict):
        for gid, rec in data.items():
            for app in rec.get("appearances", rec.get("instances", [])) or []:
                clip = str(app.get("clip_id") or app.get("clip") or app.get("video") or "")
                local = app.get("track_id") or app.get("index") or app.get("local_id")
                if clip and isinstance(local, int):
                    m[(clip, int(local))] = str(gid)
    elif isinstance(data, list):
        for item in data:
            gid = item.get("global_id") or item.get("id") or item.get("gid")
            for app in item.get("appearances", item.get("instances", [])) or []:
                clip = str(app.get("clip_id") or app.get("clip") or app.get("video") or "")
                local = app.get("track_id") or app.get("index") or app.get("local_id")
                if gid and clip and isinstance(local, int):
                    m[(clip, int(local))] = str(gid)
    return m

def _local_to_global(local_id: int, clip_id: str, id_map: Dict[Tuple[str, int], str]) -> str:
    return id_map.get((str(clip_id), int(local_id)), f"T{int(local_id)}@{clip_id}")

def _nearby_ids(sample, mid_s: float, fps: float, id_map: Dict[Tuple[str, int], str], max_ids: int = 4) -> List[str]:
    """Pick the frame nearest to mid_s and list present global IDs (if any)."""
    fnum = int(round(mid_s * fps))
    frame = sample.frames.get(fnum)
    if not frame or not getattr(frame, "detections", None):
        # Try closest existing frame
        keys = sorted(sample.frames.keys())
        if not keys:
            return []
        fnum = min(keys, key=lambda k: abs(k - fnum))
        frame = sample.frames.get(fnum)
        if not frame or not getattr(frame, "detections", None):
            return []
    dets = frame.detections.detections or []
    clip_id = Path(sample.filepath).stem
    gids = []
    for d in dets[:max_ids]:
        local = getattr(d, "index", None)
        if local is None:
            continue
        gids.append(_local_to_global(int(local), clip_id, id_map))
    return gids

# ---------------------
# Inference backends
# ---------------------
class HFVideoClassifier:
    """
    Thin wrapper around a HuggingFace VideoMAE crime/anomaly classifier.

    Works with:
      - Binary:  Nikeytas/videomae-crime-detector-ultra-v1  (Violent vs Non-Violent)  [default]  :contentReference[oaicite:5]{index=5}
      - Multi:   OPear/videomae-large-finetuned-UCF-Crime   (UCF-Crime labels)        [--model]  :contentReference[oaicite:6]{index=6}
    """
    def __init__(self, model_name: str = HF_DEFAULT_BINARY, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForVideoClassification.from_pretrained(model_name).to(self.device).eval()
        self.proc = AutoImageProcessor.from_pretrained(model_name)
        # Resolve label map
        id2label = getattr(self.model.config, "id2label", None)
        self.id2label = {int(k): v for k, v in (id2label.items() if isinstance(id2label, dict) else {})}
        self.label2id = {v: k for k, v in self.id2label.items()}
        # frames expected
        self.num_frames = int(getattr(self.model.config, "num_frames", 16))

    def classify_window(self, frames: List[np.ndarray]) -> Tuple[str, float, Dict[str, float]]:
        """
        Returns (top_label, top_prob, dist[label->prob]).
        """
        if len(frames) == 0:
            return "Unknown", 0.0, {}

        inputs = self.proc(frames[: self.num_frames], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

        dist = {self.id2label.get(i, str(i)): float(p) for i, p in enumerate(probs)}
        top_idx = int(np.argmax(probs))
        top_label = self.id2label.get(top_idx, str(top_idx))
        return top_label, float(probs[top_idx]), dist

    def is_crime(self, label: str) -> bool:
        name = (label or "").strip().lower()
        if self.model_name == HF_DEFAULT_BINARY:
            # Binary model: any positive class is crime.
            return "violent" in name or name == "1"
        # Multiclass: map to crime set
        if name == "normal videos" or name == "normal":
            return False
        if "road accident" in name:
            return bool(TREAT_ACCIDENTS_AS_CRIME)
        # If any of the violence keys is present, treat as crime
        return any(k in name for k in VIOLENCE_KEYS)

# ---------------------
# Label aggregation
# ---------------------
def _aggregate_to_scene_label(windows: List[WindowPred], binary_model: bool) -> Tuple[str, str]:
    """
    Decide final scene label + justification text from window predictions.
    For binary models: majority vote / max prob.
    For multiclass: any crime window with prob>=0.5 wins; else normal.
    """
    if not windows:
        return "normal", "No confident crime evidence predicted."

    # pick top windows
    top = sorted(windows, key=lambda w: w.score, reverse=True)[:3]
    crime_windows = [w for w in windows if w.label and w.score >= 0.50 and not w.label.lower().startswith("normal")]
    if binary_model:
        # majority over >=0.5
        votes_crime = sum(1 for w in windows if "violent" in w.label.lower() and w.score >= 0.5)
        votes_total = len(windows)
        label = "crime" if (votes_crime > votes_total / 2 or (top and "violent" in top[0].label.lower())) else "normal"
    else:
        label = "crime" if crime_windows else "normal"

    if label == "crime":
        picks = (crime_windows[:2] or top[:2])
        parts = [f"{w.label} {_format_ts(w.start_s)}–{_format_ts(w.end_s)} (p={w.score:.2f})" for w in picks]
        justification = "; ".join(parts)
    else:
        justification = "Model predicted normal/no-violence across windows."

    return label, justification

# ---------------------
# Public API
# ---------------------
def classify_scenes(
    dataset,
    all_detections=None,
    output_file: str = "scene_labels.jsonl",
    also_write_csv: bool = True,
    model_name: str = HF_DEFAULT_BINARY,
    window_sec: float = 2.0,
    stride_sec: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    SECTION B — Scene Labelling using PRETRAINED MODELS (no hand-tuned heuristics).

    For each clip:
      • samples overlapping windows (default 2s with 1s stride),
      • runs a VideoMAE classifier (Hugging Face) on each window,
      • aggregates to a single {normal|crime} label, with timestamped justification,
      • optionally references global person IDs present near crime windows.

    Outputs:
      - JSONL: one record per clip with evidence and per-window scores
      - CSV:   compact table (clip_id, label, justification)

    Change model with `model_name`:
      - Binary:  '{HF_DEFAULT_BINARY}'
      - Multi:   '{HF_DEFAULT_MULTICLASS}'
    """
    # Prepare ID mapping from Part A
    gid_map = _load_global_map("catalogue_simple.json")

    # Load model once
    clf = HFVideoClassifier(model_name=model_name)
    use_binary = (model_name == HF_DEFAULT_BINARY)

    results: List[Dict[str, Any]] = []

    for sample in dataset.iter_samples(progress=True):
        clip_id = Path(sample.filepath).stem
        path = sample.filepath
        nf, fps = _video_len_and_fps(path)
        duration = nf / fps if nf > 0 else 0.0

        if nf == 0 or duration == 0.0:
            results.append({
                "clip_id": clip_id,
                "label": "normal",
                "justification": "Empty/corrupt video.",
                "evidence": {"windows": []},
            })
            continue

        # Build windows
        windows: List[WindowPred] = []
        t = 0.0
        num_frames = clf.num_frames or 16
        while t < max(0.0, duration - 1e-6):
            start_s = t
            end_s = min(t + window_sec, duration)
            idx = _sample_indices(nf, fps, start_s, end_s, num_frames)
            frames = _read_frames(path, idx)
            label, score, dist = clf.classify_window(frames)

            # Convert multi-class label into crime/normal notion for counting
            if not use_binary and not clf.is_crime(label) and "normal" not in label.lower():
                # Re-map if a non-violent label presents (e.g., "Road Accidents") and policy says not crime.
                pass

            windows.append(WindowPred(start_s, end_s, label, score, dist))
            t += stride_sec

        # Aggregate
        final_label, justification = _aggregate_to_scene_label(windows, binary_model=use_binary)

        # Enrich justification with present IDs near the strongest crime window
        addendum = ""
        if final_label == "crime" and gid_map:
            best = max(windows, key=lambda w: (("violent" in w.label.lower()) or clf.is_crime(w.label), w.score))
            mid = 0.5 * (best.start_s + best.end_s)
            ids = _nearby_ids(sample, mid, fps, gid_map, max_ids=4)
            if ids:
                addendum = f" | people present: {', '.join(ids)}"
                justification = justification + addendum

        # Pack evidence
        evidence_windows = []
        for w in windows:
            evidence_windows.append({
                "ts": [_format_ts(w.start_s), _format_ts(w.end_s)],
                "label": w.label,
                "score": round(w.score, 4),
                "probs": {k: round(float(v), 4) for k, v in sorted(w.raw.items(), key=lambda kv: -kv[1])[:5]},
            })

        rec = {
            "clip_id": clip_id,
            "label": final_label,
            "justification": justification,
            "evidence": {
                "model": clf.model_name,
                "window_sec": window_sec,
                "stride_sec": stride_sec,
                "windows": evidence_windows,
            },
        }
        results.append(rec)

    # Write files
    outp = Path(output_file)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if also_write_csv:
        try:
            import csv
            csvp = outp.with_suffix(".csv")
            with csvp.open("w", newline="", encoding="utf-8") as cf:
                w = csv.writer(cf)
                w.writerow(["clip_id", "label", "justification"])
                for r in results:
                    w.writerow([r["clip_id"], r["label"], r["justification"]])
        except Exception as e:
            warnings.warn(f"CSV write failed: {e!r}")

    print(f"[Section B] (HF) Wrote scene labels to: {outp.resolve()}")
    return results

# ---------------------
# (Optional) AVA action detection backend stub
# If you want to use AVA Slow/SlowFast and treat actions like
# 'fight/hit (a person)', 'kick (a person)', 'push (another person)'
# as crime, wire PyTorchVideo detection here. See:
# - PyTorchVideo AVA detection tutorial and models (Slow/SlowFast)  :contentReference[oaicite:7]{index=7}
# - AVA v2.2 label list (IDs 64 fight/hit, 71 kick a person, 76 push another person)  :contentReference[oaicite:8]{index=8}
# ---------------------
