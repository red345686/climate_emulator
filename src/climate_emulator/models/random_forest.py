"""Multi-output random forest on training-only target EOF coefficients."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from climate_emulator.data.preprocessing import PreparedData, WeightedPCA
from climate_emulator.models.base import Emulator


class RandomForestEmulator(Emulator):
    """Compressed target baseline explicitly named rf_eof in provenance."""

    def __init__(self, hyperparameters: dict, components: int = 64, variance: float = .99, seed: int = 17):
        self.hyperparameters, self.components, self.variance, self.seed = hyperparameters, components, variance, seed

    def fit(self, data: PreparedData, validation: PreparedData | None = None) -> "RandomForestEmulator":
        """Fit EOF basis and forest exclusively on training endpoints."""
        self.pca = WeightedPCA(self.components, self.variance, self.seed).fit(data.target, data.weights, data.mask)
        scores = self.pca.transform(data.target)
        self.estimator = RandomForestRegressor(random_state=self.seed, bootstrap=True, **self.hyperparameters)
        self.estimator.fit(data.features, scores)
        self.training_keys = list(data.keys)
        self.diagnostics = {"implementation": "rf_eof", "retained_components": self.pca.retained, "achieved_variance": self.pca.achieved_variance, "training_reconstruction_rmse": float(self.pca.reconstruction_rmse(data.target, data.weights).mean())}
        return self

    def predict(self, data: PreparedData) -> np.ndarray:
        """Predict EOF coefficients and restore the original target grid."""
        scores = self.estimator.predict(data.features)
        if scores.ndim == 1:
            scores = scores[:, None]
        return self.pca.inverse_transform(scores)
