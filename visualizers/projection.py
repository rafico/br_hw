from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def _pca_2d(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    n_samples = features.shape[0]
    if n_samples == 1:
        return np.zeros((1, 2), dtype=np.float32)

    n_components = 2 if min(features.shape[0], features.shape[1]) >= 2 else 1
    coords = PCA(n_components=n_components, random_state=51).fit_transform(features)
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros((coords.shape[0],), dtype=np.float32)])
    return coords.astype(np.float32, copy=False)


def project_2d(features, method: str = "umap", seed: int = 51) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features[None, :]
    if features.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    normalized_method = str(method).lower()
    if normalized_method != "umap" or features.shape[0] < 3:
        return _pca_2d(features)

    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            n_neighbors=max(2, min(15, features.shape[0] - 1)),
            random_state=int(seed),
            init="spectral",
        )
        coords = reducer.fit_transform(features)
        return np.asarray(coords, dtype=np.float32)
    except Exception:
        return _pca_2d(features)
