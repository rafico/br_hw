import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import cluster_v2
import finetune_reid
import run


class _FakeProcessor:
    def __call__(self, images, return_tensors="pt"):
        pixel_values = []
        for image in images:
            mean_value = float(np.asarray(image, dtype=np.float32).mean())
            pixel_values.append(torch.full((3, 4, 4), mean_value, dtype=torch.float32))
        return {"pixel_values": torch.stack(pixel_values, dim=0)}


class _FakeClipModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def to(self, device):
        return self

    def get_image_features(self, pixel_values):
        pooled = pixel_values.mean(dim=(2, 3))
        return pooled * self.scale


class _FakeClipExtractor:
    def extract_from_detections(self, frame, boxes_xyxy):
        marker = int(np.asarray(frame, dtype=np.uint8)[0, 0, 0])
        if marker in {1, 2}:
            feature = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            feature = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        features = np.repeat(feature[None, :], len(boxes_xyxy), axis=0)
        keep_idx = np.arange(len(boxes_xyxy), dtype=np.int64)
        return features, keep_idx


class FineTuneSelectionTests(unittest.TestCase):
    def test_select_eligible_clusters_filters_on_frames_and_probability(self):
        payload = {
            "catalogue": {
                "1": [
                    {"num_frames": 20, "cluster_probability": 0.95},
                    {"num_frames": 15, "cluster_probability": 0.91},
                ],
                "2": [
                    {"num_frames": 40, "cluster_probability": 0.5},
                ],
            }
        }

        eligible = finetune_reid.select_eligible_clusters(payload)

        self.assertEqual(sorted(eligible.keys()), [1])

    def test_build_triplets_returns_empty_when_not_enough_clusters(self):
        payload = {
            "catalogue": {
                "1": [
                    {
                        "clip_id": "1",
                        "local_track_id": 1,
                        "frame_ranges": [[1, 40]],
                        "num_frames": 40,
                        "cluster_probability": 0.95,
                    }
                ],
                "2": [
                    {
                        "clip_id": "2",
                        "local_track_id": 2,
                        "frame_ranges": [[1, 40]],
                        "num_frames": 40,
                        "cluster_probability": 0.95,
                    }
                ],
            }
        }

        detections = []
        triplets = finetune_reid.build_triplets(payload, detections)

        self.assertEqual(triplets, [])


class FineTunePipelineTests(unittest.TestCase):
    def _make_detection(self, clip_id, frame_num, emb):
        return {
            "clip_id": str(clip_id),
            "video_path": f"/tmp/{clip_id}.mp4",
            "frame_num": frame_num,
            "track_id": 1,
            "embeddings": np.asarray(emb, dtype=np.float32),
            "quality": 0.95,
            "confidence": 0.95,
            "timestamp_sec": frame_num / 10.0,
            "box_xyxy_abs": [0.0, 0.0, 10.0, 30.0],
            "frame_width": 100,
            "frame_height": 100,
            "torso_hist": np.asarray([1.0, 0.0], dtype=np.float32),
            "center_x": 0.1,
            "center_y": 0.2,
            "bbox_w": 0.1,
            "bbox_h": 0.3,
        }

    def test_two_pass_smoke_train_and_recluster(self):
        class FakeKMedoids:
            def __init__(self, n_clusters, metric, random_state):
                self.n_clusters = n_clusters

            def fit(self, embeddings):
                self.medoid_indices_ = np.arange(min(self.n_clusters, len(embeddings)))
                return self

        class FakeHDBSCAN:
            def __init__(self, **kwargs):
                pass

            def fit(self, dist_matrix):
                if float(dist_matrix[0, 1]) < 0.15:
                    self.labels_ = np.array([0, 0, 1], dtype=np.int32)
                else:
                    self.labels_ = np.array([0, 1, 2], dtype=np.int32)
                self.probabilities_ = np.array([0.99, 0.99, 0.99], dtype=np.float32)
                return self

        detections = [
            self._make_detection("1", 1, [1.0, 0.0, 0.0]),
            self._make_detection("1", 2, [1.0, 0.0, 0.0]),
            self._make_detection("2", 1, [0.0, 1.0, 0.0]),
            self._make_detection("2", 2, [0.0, 1.0, 0.0]),
            self._make_detection("3", 1, [0.0, 0.0, 1.0]),
            self._make_detection("3", 2, [0.0, 0.0, 1.0]),
        ]
        video_meta = {
            "1": {"fps": 10.0, "frame_count": 100},
            "2": {"fps": 10.0, "frame_count": 100},
            "3": {"fps": 10.0, "frame_count": 100},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            catalogue_path = Path(tmpdir) / "catalogue_v2.json"
            output_weights = Path(tmpdir) / "clipreid_ft.pth"

            with mock.patch.object(cluster_v2, "_get_clustering_backends", return_value=(FakeKMedoids, FakeHDBSCAN)):
                pass1 = cluster_v2.generate_person_catalogue_v2(
                    detections,
                    video_meta=video_meta,
                    output_file=str(catalogue_path),
                    top_k_frames=2,
                    n_prototypes=1,
                    use_rerank=False,
                    cooccurrence_constraint=True,
                )

                with mock.patch.object(finetune_reid.CLIPImageProcessor, "from_pretrained", return_value=_FakeProcessor()), \
                        mock.patch.object(finetune_reid.CLIPModel, "from_pretrained", return_value=_FakeClipModel()), \
                        mock.patch.object(
                            finetune_reid,
                            "_load_crop",
                            side_effect=lambda det: np.full((8, 8, 3), int(det["clip_id"]), dtype=np.uint8),
                        ):
                    weights_path = finetune_reid.train(
                        detections=detections,
                        catalogue_path=str(catalogue_path),
                        output_weights=str(output_weights),
                        epochs=1,
                        batch_size=2,
                        min_frames=2,
                        min_probability=0.9,
                    )

                self.assertIsNotNone(weights_path)
                self.assertTrue(output_weights.exists())

                with mock.patch.object(run, "load_reid_extractor", return_value=_FakeClipExtractor()), \
                        mock.patch.object(
                            run,
                            "_load_rgb_frame_for_reembedding",
                            side_effect=lambda video_path, frame_num: np.full(
                                (8, 8, 3),
                                int(Path(video_path).stem),
                                dtype=np.uint8,
                            ),
                        ):
                    finetuned_detections = run.reembed_detections_with_finetuned_clip(
                        detections,
                        reid_backbone="clipreid",
                        clip_weights_path=str(output_weights),
                    )

                pass2_path = Path(tmpdir) / "catalogue_v2_pass2.json"
                pass2 = cluster_v2.generate_person_catalogue_v2(
                    finetuned_detections,
                    video_meta=video_meta,
                    output_file=str(pass2_path),
                    top_k_frames=2,
                    n_prototypes=1,
                    use_rerank=False,
                    cooccurrence_constraint=True,
                )

        self.assertEqual(pass1["summary"]["total_unique_persons"], 3)
        self.assertLessEqual(
            pass2["summary"]["total_unique_persons"],
            pass1["summary"]["total_unique_persons"],
        )
        self.assertEqual(pass2["summary"]["total_tracklets"], 3)


if __name__ == "__main__":
    unittest.main()
