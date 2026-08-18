from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class PCABaseline:
    """
    Optional PCA baseline.

    This module is deliberately not used by the main P3
    preprocessing pipeline. It exists for later benchmarking.
    """

    def __init__(self, n_components: int = 2):
        self.model = PCA(n_components=n_components)

    def fit(self, X: np.ndarray):
        X = np.asarray(X)

        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        self.model.fit(X)
        return self

    def transform(self, X: np.ndarray):
        X = np.asarray(X)

        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        return self.model.transform(X)

    def inverse_transform(self, X: np.ndarray):
        return self.model.inverse_transform(X)