"""Two-stage cumulative-emissions/global-temperature linear pattern scaling."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from climate_emulator.data.preprocessing import PreparedData
from climate_emulator.models.base import Emulator, restore_field


class PatternScaling(Emulator):
    """OLS CO2 -> global tas -> local target; no observed test tas is used."""

    def fit(self, data: PreparedData, validation: PreparedData | None = None) -> "PatternScaling":
        """Estimate response patterns against observed training global temperature."""
        temperature_mask = np.isfinite(data.temperature).all(axis=0) & (data.weights > 0)
        if not temperature_mask.any():
            raise ValueError("No finite training global-temperature cells")
        w = data.weights[temperature_mask]
        global_temperature = data.temperature[:, temperature_mask] @ (w / w.sum())
        co2 = data.support[:, 0:1]
        if np.ptp(co2) <= 1e-12 or np.ptp(global_temperature) <= 1e-12:
            raise ValueError("LPS global regression is unidentified on constant training values")
        self.global_model = LinearRegression().fit(co2, global_temperature)
        self.mask = data.mask.copy()
        self.local_model = LinearRegression().fit(global_temperature[:, None], data.target[:, self.mask])
        self.training_keys = list(data.keys)
        self.diagnostics = {"global_co2_slope": float(self.global_model.coef_[0]), "global_intercept": float(self.global_model.intercept_), "temperature_cells": int(temperature_mask.sum())}
        return self

    def predict(self, data: PreparedData) -> np.ndarray:
        """Use only emissions to predict the global driver at inference."""
        global_temperature = self.global_model.predict(data.support[:, 0:1])
        return restore_field(self.local_model.predict(global_temperature[:, None]), self.mask)
