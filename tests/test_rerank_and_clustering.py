import unittest

import numpy as np

from generate_person_catalogue import (
    _cluster_from_sorted_pairs,
    build_distance_matrix,
    tracklets_cooccur,
)
from rerank import kreciprocal_rerank


def _tracklet(clip_id, track_id, start, end, emb):
    emb = np.asarray(emb, dtype=np.float32)
    emb = emb / np.linalg.norm(emb)
    return {
        "clip_id": str(clip_id),
        "track_id": int(track_id),
        "embedding": emb,
        "all_embeddings": np.stack([emb], axis=0),
        "frame_ranges": [[start, end]],
        "num_frames": end - start + 1,
    }


class RerankTests(unittest.TestCase):
    def test_rerank_returns_symmetric_nonnegative_matrix(self):
        rng = np.random.default_rng(51)
        feats = rng.normal(size=(20, 128)).astype(np.float32)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)

        dist = kreciprocal_rerank(feats)

        self.assertEqual(dist.shape, (20, 20))
        self.assertEqual(dist.dtype, np.float32)
        self.assertTrue(np.allclose(dist, dist.T, atol=1e-6))
        self.assertTrue(np.all(dist >= 0.0))
        self.assertTrue(np.allclose(np.diag(dist), 0.0))

    def test_build_distance_matrix_supports_rerank_path(self):
        tracklets = [
            _tracklet("1", 1, 1, 10, [1.0, 0.0, 0.0]),
            _tracklet("2", 2, 1, 10, [0.99, 0.01, 0.0]),
            _tracklet("3", 3, 1, 10, [0.0, 1.0, 0.0]),
        ]

        dist = build_distance_matrix(tracklets, linkage="min", use_rerank=True)

        self.assertEqual(dist.shape, (3, 3))
        self.assertTrue(np.allclose(dist, dist.T, atol=1e-6))
        self.assertLess(dist[0, 1], dist[0, 2])


class CooccurrenceConstraintTests(unittest.TestCase):
    def test_tracklets_cooccur_detects_same_clip_overlap_only(self):
        a = _tracklet("2", 1, 1, 10, [1.0, 0.0, 0.0])
        b = _tracklet("2", 2, 8, 20, [1.0, 0.0, 0.0])
        c = _tracklet("2", 3, 21, 30, [1.0, 0.0, 0.0])
        d = _tracklet("3", 4, 8, 20, [1.0, 0.0, 0.0])

        self.assertTrue(tracklets_cooccur(a, b))
        self.assertFalse(tracklets_cooccur(a, c))
        self.assertFalse(tracklets_cooccur(a, d))

    def test_cooccurrence_constraint_allows_non_overlapping_same_clip_merge(self):
        tracklets = [
            _tracklet("2", 1, 1, 10, [1.0, 0.0, 0.0]),
            _tracklet("2", 2, 11, 20, [1.0, 0.0, 0.0]),
        ]
        labels = _cluster_from_sorted_pairs(
            sorted_pairs=[(0.01, 0, 1)],
            tracklet_info=tracklets,
            min_cluster_size=2,
            cluster_selection_epsilon=0.1,
            cooccurrence_constraint=True,
        )

        self.assertTrue(np.array_equal(labels, np.array([0, 0], dtype=np.int32)))

    def test_cooccurrence_constraint_blocks_overlapping_same_clip_merge(self):
        tracklets = [
            _tracklet("2", 1, 1, 10, [1.0, 0.0, 0.0]),
            _tracklet("2", 2, 8, 20, [1.0, 0.0, 0.0]),
        ]
        labels = _cluster_from_sorted_pairs(
            sorted_pairs=[(0.01, 0, 1)],
            tracklet_info=tracklets,
            min_cluster_size=2,
            cluster_selection_epsilon=0.1,
            cooccurrence_constraint=True,
        )

        self.assertTrue(np.array_equal(labels, np.array([-1, -1], dtype=np.int32)))

    def test_legacy_cross_clip_only_mode_rejects_same_clip_merge(self):
        tracklets = [
            _tracklet("2", 1, 1, 10, [1.0, 0.0, 0.0]),
            _tracklet("2", 2, 11, 20, [1.0, 0.0, 0.0]),
        ]
        labels = _cluster_from_sorted_pairs(
            sorted_pairs=[(0.01, 0, 1)],
            tracklet_info=tracklets,
            min_cluster_size=2,
            cluster_selection_epsilon=0.1,
            cooccurrence_constraint=False,
        )

        self.assertTrue(np.array_equal(labels, np.array([-1, -1], dtype=np.int32)))


if __name__ == "__main__":
    unittest.main()
