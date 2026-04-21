from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPModel

from reid_model import DetectionReIDExtractor

LOGGER = logging.getLogger(__name__)
_CLIP_MODEL_ID = "openai/clip-vit-base-patch16"


def _normalize_rows(feats: np.ndarray) -> np.ndarray:
    feats = np.asarray(feats, dtype=np.float32)
    if feats.size == 0:
        return feats.astype(np.float32, copy=False)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.where(norms == 0, 1.0, norms)


class CLIPReIDExtractor:
    """Same API as DetectionReIDExtractor using a CLIP ViT-B/16 image encoder."""

    def __init__(
        self,
        model_path: str = "",
        image_size: Tuple[int, int] = (256, 128),
        device: str = "cuda",
        batch_size: int = 32,
    ):
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.image_size = tuple(int(v) for v in image_size)
        self.processor = CLIPImageProcessor.from_pretrained(_CLIP_MODEL_ID)
        self.model = CLIPModel.from_pretrained(_CLIP_MODEL_ID).to(self.device)
        self.model.eval()

        if model_path:
            weight_path = Path(model_path)
            if weight_path.exists():
                state_dict = torch.load(weight_path, map_location=self.device)
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                if missing:
                    LOGGER.warning(
                        "CLIPReIDExtractor loaded %s with missing keys: %s",
                        weight_path,
                        sorted(missing)[:5],
                    )
                if unexpected:
                    LOGGER.warning(
                        "CLIPReIDExtractor loaded %s with unexpected keys: %s",
                        weight_path,
                        sorted(unexpected)[:5],
                    )
            else:
                LOGGER.warning("CLIP-ReID weights path does not exist: %s", weight_path)

    @staticmethod
    def _clip_boxes(boxes: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        boxes = boxes.copy()
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes, valid

    def extract_from_detections(
        self,
        frame: np.ndarray,
        boxes_xyxy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if boxes_xyxy.size == 0:
            return np.empty((0, 512), dtype=np.float32), np.empty((0,), dtype=np.int64)

        h, w = frame.shape[:2]
        boxes, valid = self._clip_boxes(boxes_xyxy, h, w)
        keep_idx = np.flatnonzero(valid)
        if keep_idx.size == 0:
            return np.empty((0, 512), dtype=np.float32), keep_idx

        crops: List[np.ndarray] = []
        target_h, target_w = self.image_size
        for idx in keep_idx:
            x1, y1, x2, y2 = boxes[idx]
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue
            crops.append(crop)

        if not crops:
            return np.empty((0, 512), dtype=np.float32), np.empty((0,), dtype=np.int64)

        all_feats: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                batch = crops[start:start + self.batch_size]
                resized_batch = [
                    cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    for crop in batch
                ]
                inputs = self.processor(images=resized_batch, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                feats = self.model.get_image_features(pixel_values=pixel_values)
                all_feats.append(feats.detach().cpu().numpy().astype(np.float32))

        feats_all = np.concatenate(all_feats, axis=0) if all_feats else np.empty((0, 512), dtype=np.float32)
        feats_all = _normalize_rows(feats_all)
        return feats_all, keep_idx[: len(feats_all)]


class EnsembleExtractor:
    """Concatenates L2-normalized feats from multiple extractors; re-normalizes."""

    def __init__(self, extractors: List[object]):
        self.extractors = list(extractors)

    def extract_from_detections(self, frame, boxes_xyxy):
        if not self.extractors:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

        extractor_outputs = [
            extractor.extract_from_detections(frame, boxes_xyxy)
            for extractor in self.extractors
        ]

        feature_maps: List[Dict[int, np.ndarray]] = []
        common_indices = None
        for feats, keep_idx in extractor_outputs:
            feats = _normalize_rows(feats)
            mapping = {
                int(idx): feats[row]
                for row, idx in enumerate(np.asarray(keep_idx, dtype=np.int64))
            }
            feature_maps.append(mapping)
            idx_set = set(mapping)
            common_indices = idx_set if common_indices is None else (common_indices & idx_set)

        if not common_indices:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

        ordered_idx = np.array(sorted(common_indices), dtype=np.int64)
        concatenated = []
        for idx in ordered_idx:
            parts = [mapping[int(idx)] for mapping in feature_maps]
            concatenated.append(np.concatenate(parts, axis=0))

        feats = _normalize_rows(np.asarray(concatenated, dtype=np.float32))
        return feats, ordered_idx


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = max(area_a + area_b - inter, 1e-12)
    return float(inter / union)


def filter_crops_for_reid(
    boxes_xyxy: np.ndarray,
    frame_shape: tuple,
    other_boxes: np.ndarray,
    use_pose: bool = False,
    pose_model=None,
) -> np.ndarray:
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
    other_boxes = np.asarray(other_boxes, dtype=np.float32)
    if boxes_xyxy.size == 0:
        return np.zeros((0,), dtype=bool)

    frame_h, frame_w = frame_shape[:2]
    frame_area = max(float(frame_h * frame_w), 1.0)
    keep = np.ones((boxes_xyxy.shape[0],), dtype=bool)

    clipped = boxes_xyxy.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, max(frame_w - 1, 0))
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, max(frame_h - 1, 0))

    for idx, box in enumerate(clipped):
        x1, y1, x2, y2 = box
        width = max(float(x2 - x1), 0.0)
        height = max(float(y2 - y1), 0.0)
        if width <= 0.0 or height <= 0.0:
            keep[idx] = False
            continue

        aspect_ratio = height / max(width, 1e-6)
        if not 1.3 <= aspect_ratio <= 4.0:
            keep[idx] = False
            continue

        area = width * height
        if area < 0.0015 * frame_area:
            keep[idx] = False
            continue

        max_iou = 0.0
        for other in other_boxes:
            if np.allclose(other, boxes_xyxy[idx], atol=1e-5):
                continue
            max_iou = max(max_iou, _box_iou(box, other))
        if max_iou >= 0.55:
            keep[idx] = False
            continue

        if use_pose:
            visible_keypoints = 0
            if callable(pose_model):
                visible_keypoints = int(pose_model(box))
            if visible_keypoints < 7:
                keep[idx] = False

    return keep


def build_extractor(
    name: str,
    device: str,
    image_size=(256, 128),
    batch_size=32,
    model_path: str = "",
    input_is_bgr: bool = False,
):
    """Build a configured detector-level ReID extractor."""
    normalized_name = str(name).lower()
    if normalized_name in {"osnet_ain", "osnet_ain_x1_0", "osnet"}:
        return DetectionReIDExtractor(
            model_name="osnet_ain_x1_0",
            model_path=model_path,
            image_size=image_size,
            device=device,
            batch_size=batch_size,
            input_is_bgr=input_is_bgr,
        )
    if normalized_name == "clipreid":
        return CLIPReIDExtractor(
            model_path=model_path,
            image_size=image_size,
            device=device,
            batch_size=batch_size,
        )
    if normalized_name == "ensemble":
        return EnsembleExtractor(
            [
                build_extractor(
                    "osnet_ain",
                    device=device,
                    image_size=image_size,
                    batch_size=batch_size,
                    model_path="",
                    input_is_bgr=input_is_bgr,
                ),
                build_extractor(
                    "clipreid",
                    device=device,
                    image_size=image_size,
                    batch_size=batch_size,
                    model_path=model_path,
                    input_is_bgr=input_is_bgr,
                ),
            ]
        )
    raise ValueError(f"Unsupported extractor '{name}'. Expected osnet_ain, clipreid, or ensemble.")
