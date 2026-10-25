"""Standardized multi-forcer multi-output ridge baseline."""
from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from climate_emulator.data.preprocessing import PreparedData
from climate_emulator.models.base import Emulator, restore_field


class RidgeEmulator(Emulator):
    """One frozen feature scaler and multi-output ridge model per fitted run."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, data: PreparedData, validation: PreparedData | None = None) -> "RidgeEmulator":
        """Fit on training-only PCA features and valid target cells."""
        self.mask = data.mask.copy()
        self.estimator = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha, solver="svd"))
        self.estimator.fit(data.features, data.target[:, self.mask])
        self.training_keys = list(data.keys)
        return self

    def predict(self, data: PreparedData) -> np.ndarray:
        """Reconstruct predictions in physical target units."""
        return restore_field(self.estimator.predict(data.features), self.mask)
