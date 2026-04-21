from __future__ import annotations

import numpy as np


def _validate_inputs(feats: np.ndarray) -> np.ndarray:
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError("feats must be a 2D array of shape (N, D)")
    return feats


def kreciprocal_rerank(
    feats: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """Compute a symmetric re-ranked distance matrix using k-reciprocal encoding."""
    feats = _validate_inputs(feats)
    all_num = feats.shape[0]
    if all_num == 0:
        return np.empty((0, 0), dtype=np.float32)
    if all_num == 1:
        return np.zeros((1, 1), dtype=np.float32)

    k1 = max(1, min(int(k1), all_num - 1))
    k2 = max(1, min(int(k2), all_num))
    lambda_value = float(np.clip(lambda_value, 0.0, 1.0))

    similarity = np.matmul(feats, feats.T)
    original_dist = np.clip(2.0 - (2.0 * similarity), 0.0, None).astype(np.float32)
    col_max = np.maximum(original_dist.max(axis=0, keepdims=True), 1e-12)
    original_dist = (original_dist / col_max).T
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)

    V = np.zeros_like(original_dist, dtype=np.float32)
    half_k1 = max(1, int(np.around(k1 / 2)))

    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, : min(k1 + 1, all_num)]
        backward_k_neigh_index = initial_rank[
            forward_k_neigh_index, : min(k1 + 1, all_num)
        ]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index.copy()

        for candidate in k_reciprocal_index:
            candidate_forward = initial_rank[candidate, : min(half_k1 + 1, all_num)]
            candidate_backward = initial_rank[
                candidate_forward, : min(half_k1 + 1, all_num)
            ]
            fi_candidate = np.where(candidate_backward == candidate)[0]
            candidate_reciprocal = candidate_forward[fi_candidate]
            if candidate_reciprocal.size == 0:
                continue
            overlap = np.intersect1d(
                candidate_reciprocal,
                k_reciprocal_index,
                assume_unique=False,
            )
            if overlap.size > (2.0 / 3.0) * candidate_reciprocal.size:
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index,
                    candidate_reciprocal,
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weights = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        denom = max(float(np.sum(weights)), 1e-12)
        V[i, k_reciprocal_expansion_index] = weights / denom

    if k2 > 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe

    inv_index = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(all_num):
        temp_min = np.zeros((all_num,), dtype=np.float32)
        ind_non_zero = np.where(V[i, :] != 0)[0]
        for column_idx in ind_non_zero:
            shared_rows = inv_index[column_idx]
            temp_min[shared_rows] += np.minimum(
                V[i, column_idx],
                V[shared_rows, column_idx],
            )
        jaccard_dist[i] = 1.0 - (
            temp_min / np.maximum(2.0 - temp_min, 1e-12)
        )

    final_dist = (jaccard_dist * (1.0 - lambda_value)) + (
        original_dist * lambda_value
    )
    final_dist = 0.5 * (final_dist + final_dist.T)
    np.fill_diagonal(final_dist, 0.0)
    final_dist = np.clip(final_dist, 0.0, None).astype(np.float32, copy=False)
    return final_dist
