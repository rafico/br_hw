from typing import Tuple, List
import cv2
import numpy as np
from torchreid.reid.utils import FeatureExtractor


def torso_color_hist(frame_rgb: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    """8x4x3 HSV histogram over the vertical 35-65% band of the box. L1-normalized (96-d)."""
    if frame_rgb.size == 0:
        return np.zeros((96,), dtype=np.float32)

    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float32)
    x1i = int(np.clip(np.floor(x1), 0, max(w - 1, 0)))
    y1i = int(np.clip(np.floor(y1), 0, max(h - 1, 0)))
    x2i = int(np.clip(np.ceil(x2), 0, w))
    y2i = int(np.clip(np.ceil(y2), 0, h))
    if x2i <= x1i or y2i <= y1i:
        return np.zeros((96,), dtype=np.float32)

    band_top = y1i + int(0.35 * max(y2i - y1i, 1))
    band_bottom = y1i + int(0.65 * max(y2i - y1i, 1))
    band_top = int(np.clip(band_top, y1i, max(y2i - 1, y1i)))
    band_bottom = int(np.clip(max(band_bottom, band_top + 1), band_top + 1, y2i))
    crop = frame_rgb[band_top:band_bottom, x1i:x2i]
    if crop.size == 0:
        return np.zeros((96,), dtype=np.float32)

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [8, 4, 3],
        [0, 180, 0, 256, 0, 256],
    ).astype(np.float32)
    hist = hist.reshape(-1)
    total = float(np.sum(hist))
    if total <= 0:
        return np.zeros((96,), dtype=np.float32)
    return hist / total


def torso_color_chi2(h1: np.ndarray, h2: np.ndarray) -> float:
    """Symmetric chi-squared distance in [0, 2]."""
    a = np.asarray(h1, dtype=np.float32).reshape(-1)
    b = np.asarray(h2, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 2.0
    numer = np.square(a - b)
    denom = np.maximum(a + b, 1e-12)
    dist = float(np.sum(numer / denom))
    return float(np.clip(dist, 0.0, 2.0))

class DetectionReIDExtractor:
    """
    Extracts ReID features for person detections from full frames.

    Args:
        model_name: torchreid model name (e.g. "osnet_ain_x1_0")
        model_path: optional path to fine-tuned checkpoint
        image_size: (H, W) for the ReID model
        device: "cuda" or "cpu"
        batch_size: how many crops to process at once
        input_is_bgr: set True if you pass BGR frames; otherwise RGB is assumed
    """
    def __init__(
        self,
        model_name: str = "osnet_ain_x1_0",
        model_path: str = "",
        image_size: Tuple[int, int] = (256, 128),
        device: str = "cuda",
        batch_size: int = 32,
        input_is_bgr: bool = False,
    ):
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            image_size=image_size,
            device=device,
        )
        self.batch_size = max(1, int(batch_size))
        self.input_is_bgr = input_is_bgr

    @staticmethod
    def _clip_boxes(boxes: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Clip xyxy boxes to image bounds; return clipped boxes and valid mask."""
        boxes = boxes.copy()
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        # valid if area > 0
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes, valid

    def extract_from_detections(
        self,
        frame: np.ndarray,                  # HxWx3; RGB by default (see input_is_bgr)
        boxes_xyxy: np.ndarray,             # Nx4 floats (x1, y1, x2, y2)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            feats: (M, D) float32 array of features for valid boxes (M <= N)
            keep_idx: indices into the original 'boxes_xyxy' that were valid
        """
        if boxes_xyxy.size == 0:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

        if self.input_is_bgr:
            # convert BGR->RGB for torchreid
            frame_rgb = frame[:, :, ::-1]
        else:
            frame_rgb = frame

        h, w = frame_rgb.shape[:2]
        boxes, valid = self._clip_boxes(boxes_xyxy, h, w)
        keep_idx = np.flatnonzero(valid)

        if keep_idx.size == 0:
            return np.empty((0, 0), dtype=np.float32), keep_idx

        # Crop persons
        crops: List[np.ndarray] = []
        for i in keep_idx:
            x1, y1, x2, y2 = boxes[i]
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            crop = frame_rgb[y1i:y2i, x1i:x2i]
            # torchreid can take raw uint8 HxWx3 arrays and will resize/normalize internally
            crops.append(crop)

        # Batch through torchreid extractor to control memory (it accepts lists)
        feats_list: List[np.ndarray] = []
        for s in range(0, len(crops), self.batch_size):
            batch = crops[s:s + self.batch_size]
            # extractor returns (B, D) numpy array
            feats = self.extractor(batch)  # type: ignore
            # Ensure numpy float32
            feats_list.append(np.asarray(feats.cpu(), dtype=np.float32))

        feats_all = np.concatenate(feats_list, axis=0) if feats_list else np.empty((0, 0), dtype=np.float32)
        return feats_all, keep_idx
