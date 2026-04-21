import unittest

import finetune_reid


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


if __name__ == "__main__":
    unittest.main()
