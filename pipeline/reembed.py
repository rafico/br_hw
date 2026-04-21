from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import torch

from pipeline.vision import load_reid_extractor


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def _load_rgb_frame_for_reembedding(video_path: str, frame_num: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(frame_num) - 1, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _merge_clip_embedding(existing_embedding, clip_embedding: np.ndarray, reid_backbone: str) -> np.ndarray:
    clip_embedding = _normalize_rows(np.asarray(clip_embedding, dtype=np.float32)[None, :])[0]
    normalized_backbone = str(reid_backbone).lower()
    if normalized_backbone == "clipreid":
        return clip_embedding.astype(np.float32, copy=False)

    if normalized_backbone != "ensemble":
        raise ValueError(
            f"Fine-tune re-embedding only supports clipreid or ensemble backbones, got {reid_backbone!r}"
        )

    existing = np.asarray(existing_embedding, dtype=np.float32).reshape(-1)
    if existing.size <= clip_embedding.size:
        return clip_embedding.astype(np.float32, copy=False)

    osnet_part = existing[:-clip_embedding.size]
    osnet_part = _normalize_rows(osnet_part[None, :])[0]
    merged = np.concatenate([osnet_part, clip_embedding], axis=0)
    return _normalize_rows(merged[None, :])[0]


def reembed_detections_with_finetuned_clip(
        all_detections,
        *,
        reid_backbone: str,
        clip_weights_path: str,
        device: Optional[str] = None,
        extractor_loader=load_reid_extractor,
):
    if not all_detections:
        return []

    if str(reid_backbone).lower() not in {"clipreid", "ensemble"}:
        raise ValueError("--finetune-reid requires --reid-backbone clipreid or ensemble")

    clip_extractor = extractor_loader(
        model_name="clipreid",
        model_path=clip_weights_path,
        device=device,
    )

    updated = [dict(det) for det in all_detections]
    grouped_indices = defaultdict(list)
    for idx, det in enumerate(all_detections):
        grouped_indices[(det["video_path"], str(det["clip_id"]), int(det["frame_num"]))].append(idx)

    with torch.inference_mode():
        for (video_path, _clip_id, frame_num), indices in sorted(grouped_indices.items()):
            frame_rgb = _load_rgb_frame_for_reembedding(video_path, frame_num)
            if frame_rgb is None:
                continue

            boxes = np.asarray([all_detections[idx]["box_xyxy_abs"] for idx in indices], dtype=np.float32)
            features, keep_idx = clip_extractor.extract_from_detections(frame_rgb, boxes)
            feature_map = {
                int(det_offset): np.asarray(features[row], dtype=np.float32)
                for row, det_offset in enumerate(np.asarray(keep_idx, dtype=np.int64))
            }

            for det_offset, det_idx in enumerate(indices):
                clip_feature = feature_map.get(int(det_offset))
                if clip_feature is None:
                    continue
                updated[det_idx]["embeddings"] = _merge_clip_embedding(
                    all_detections[det_idx]["embeddings"],
                    clip_feature,
                    reid_backbone=reid_backbone,
                )

    return updated


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
