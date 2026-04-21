from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPModel, get_cosine_schedule_with_warmup

from clustering.common import tracklets_cooccur
from utils_determinism import seed_everything

LOGGER = logging.getLogger(__name__)


def load_catalogue(catalogue_path: str = "catalogue_v2.json") -> dict:
    with open(catalogue_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_detections(detections_path: str) -> List[dict]:
    with open(detections_path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_eligible_clusters(
    catalogue_payload: dict,
    min_frames: int = 30,
    min_probability: float = 0.9,
) -> Dict[int, List[dict]]:
    eligible = {}
    for global_id, appearances in catalogue_payload.get("catalogue", {}).items():
        total_frames = sum(int(app.get("num_frames", 0)) for app in appearances)
        probabilities = [float(app.get("cluster_probability", 0.0)) for app in appearances]
        if total_frames >= min_frames and probabilities and all(prob >= min_probability for prob in probabilities):
            eligible[int(global_id)] = appearances
    return eligible


def _group_detections_by_tracklet(detections: List[dict]) -> Dict[Tuple[str, int], List[dict]]:
    grouped: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for det in detections:
        grouped[(str(det["clip_id"]), int(det["track_id"]))].append(det)
    for key in grouped:
        grouped[key].sort(key=lambda det: int(det["frame_num"]))
    return grouped


def _best_detection(detections: List[dict]) -> Optional[dict]:
    if not detections:
        return None
    return max(
        detections,
        key=lambda det: float(det.get("quality") or det.get("confidence") or 0.0),
    )


def _farthest_pair(detections: List[dict]) -> Optional[Tuple[dict, dict]]:
    if len(detections) < 2:
        return None
    return detections[0], detections[-1]


def build_triplets(
    catalogue_payload: dict,
    detections: List[dict],
    min_frames: int = 30,
    min_probability: float = 0.9,
) -> List[Tuple[dict, dict, dict]]:
    eligible = select_eligible_clusters(
        catalogue_payload,
        min_frames=min_frames,
        min_probability=min_probability,
    )
    if len(eligible) < 3:
        return []

    grouped = _group_detections_by_tracklet(detections)
    appearances_by_gid = {
        int(gid): apps
        for gid, apps in eligible.items()
    }

    triplets = []
    for gid, appearances in appearances_by_gid.items():
        negative_appearances = [
            (other_gid, appearance)
            for other_gid, other_apps in appearances_by_gid.items()
            if other_gid != gid
            for appearance in other_apps
        ]
        for appearance in appearances:
            key = (str(appearance["clip_id"]), int(appearance["local_track_id"]))
            anchor_tracklet = grouped.get(key, [])
            anchor = _best_detection(anchor_tracklet)
            if anchor is None:
                continue

            positive = None
            for other in appearances:
                other_key = (str(other["clip_id"]), int(other["local_track_id"]))
                if other_key == key:
                    continue
                positive = _best_detection(grouped.get(other_key, []))
                if positive is not None:
                    break
            if positive is None:
                pair = _farthest_pair(anchor_tracklet)
                if pair is None:
                    continue
                anchor, positive = pair

            negative = None
            for _, negative_app in negative_appearances:
                if str(negative_app["clip_id"]) != str(appearance["clip_id"]):
                    continue
                if tracklets_cooccur(
                    {
                        "clip_id": appearance["clip_id"],
                        "frame_ranges": appearance["frame_ranges"],
                    },
                    {
                        "clip_id": negative_app["clip_id"],
                        "frame_ranges": negative_app["frame_ranges"],
                    },
                ):
                    continue
                negative_key = (str(negative_app["clip_id"]), int(negative_app["local_track_id"]))
                negative = _best_detection(grouped.get(negative_key, []))
                if negative is not None:
                    break

            if negative is None:
                for _, negative_app in negative_appearances:
                    negative_key = (str(negative_app["clip_id"]), int(negative_app["local_track_id"]))
                    negative = _best_detection(grouped.get(negative_key, []))
                    if negative is not None:
                        break

            if negative is not None:
                triplets.append((anchor, positive, negative))

    return triplets


def _load_crop(det: dict) -> np.ndarray:
    cap = cv2.VideoCapture(det["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(det["frame_num"]) - 1, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to load frame {det['frame_num']} from {det['video_path']}")
    x1, y1, x2, y2 = [int(v) for v in det["box_xyxy_abs"]]
    crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
    if crop.size == 0:
        raise RuntimeError(f"Empty crop for detection {det}")
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


class TripletDataset(Dataset):
    def __init__(self, triplets: List[Tuple[dict, dict, dict]], processor: CLIPImageProcessor):
        self.triplets = triplets
        self.processor = processor

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int):
        anchor, positive, negative = self.triplets[index]
        images = [_load_crop(anchor), _load_crop(positive), _load_crop(negative)]
        processed = self.processor(images=images, return_tensors="pt")
        return processed["pixel_values"][0], processed["pixel_values"][1], processed["pixel_values"][2]


def train(
    detections_path: str = "",
    detections: Optional[List[dict]] = None,
    catalogue_path: str = "catalogue_v2.json",
    output_weights: str = "checkpoints/clipreid_ft.pth",
    epochs: int = 5,
    batch_size: int = 64,
    margin: float = 0.3,
    learning_rate: float = 3.5e-5,
    min_frames: int = 30,
    min_probability: float = 0.9,
    seed: int = 51,
) -> Optional[Path]:
    seed_everything(seed)
    catalogue_payload = load_catalogue(catalogue_path)
    if detections is None:
        if not detections_path:
            raise ValueError("Either detections_path or detections must be provided")
        detections = load_detections(detections_path)

    triplets = build_triplets(
        catalogue_payload,
        detections,
        min_frames=min_frames,
        min_probability=min_probability,
    )
    if not triplets:
        LOGGER.warning("Fewer than 3 high-confidence clusters are available; skipping fine-tuning.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    model.train()

    dataset = TripletDataset(triplets, processor)
    dataloader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_steps = max(int(epochs * len(dataloader)), 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(total_steps // 10, 1),
        num_training_steps=total_steps,
    )
    loss_fn = nn.TripletMarginLoss(margin=margin)

    for _ in range(int(epochs)):
        for anchor_px, positive_px, negative_px in dataloader:
            anchor_px = anchor_px.to(device)
            positive_px = positive_px.to(device)
            negative_px = negative_px.to(device)
            anchor_feat = model.get_image_features(pixel_values=anchor_px)
            positive_feat = model.get_image_features(pixel_values=positive_px)
            negative_feat = model.get_image_features(pixel_values=negative_px)
            loss = loss_fn(anchor_feat, positive_feat, negative_feat)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

    output_path = Path(output_weights)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    LOGGER.info("Saved fine-tuned CLIP weights to %s", output_path.resolve())
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Pseudo-label fine-tune CLIP ReID")
    parser.add_argument("--detections-cache", required=True, help="Path to cached all_detections JSON")
    parser.add_argument("--catalogue-path", default="catalogue_v2.json")
    parser.add_argument("--output-weights", default="checkpoints/clipreid_ft.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--min-probability", type=float, default=0.9)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    train(
        detections_path=args.detections_cache,
        catalogue_path=args.catalogue_path,
        output_weights=args.output_weights,
        epochs=args.epochs,
        min_frames=args.min_frames,
        min_probability=args.min_probability,
    )


if __name__ == "__main__":
    main()
