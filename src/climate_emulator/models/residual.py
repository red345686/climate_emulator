"""Frozen LPS priors with unbounded, bounded and deterministic gated corrections."""
from __future__ import annotations

import numpy as np

from climate_emulator.data.preprocessing import PreparedData
from climate_emulator.data.support import gate
from climate_emulator.models.cnn_lstm import CNNLSTMEmulator, build_backbone
from climate_emulator.models.pattern_scaling import PatternScaling


class ResidualEmulator(CNNLSTMEmulator):
    """Bounded and gated outputs share the same fitted residual checkpoint."""

    def __init__(self, config: dict, seed: int, bounded: bool = True, epochs: int | None = None, log_directory=None):
        super().__init__(config, seed, epochs, log_directory)
        self.bounded = bounded
        self.variant = "residual_bounded" if bounded else "residual_unbounded"

    def fit(self, data: PreparedData, validation: PreparedData | None = None) -> "ResidualEmulator":
        """Freeze the training-only prior, derive the bound and fit residual weights."""
        from climate_emulator.training.trainer import train_neural
        from climate_emulator.utils import set_seed
        set_seed(self.seed, tensorflow=True, deterministic=self.config["training"]["deterministic"])
        self.mask, self.shape, self.scale = data.mask.copy(), data.target.shape[1:], data.scale
        self.prior = PatternScaling().fit(data)
        residual = data.target[:, self.mask] - self.prior.predict(data)[:, self.mask]
        w = data.weights[self.mask] / data.weights[self.mask].sum()
        residual_rms = float(np.sqrt(np.mean(np.sum(residual ** 2 * w, axis=1))))
        floor = self.config["preprocessing"]["scale_floor"]
        multiplier = self.config["models"]["residual_bounded"]["bound_multiplier"]
        self.bound = max(multiplier * residual_rms, floor)
        self.residual_rms = residual_rms
        self.network = build_backbone((data.history.indices.shape[1], *self.shape, 4), self.config["models"]["cnn_lstm"], self.config["training"]["kernel_l2"], zero_output=True)
        self.history = train_neural(self, data, validation, self.config["training"], self.epochs, self.log_directory)
        self.selected_epochs, self.training_keys = self.history["best_epoch"], list(data.keys)
        return self

    def training_target(self, data: PreparedData) -> np.ndarray:
        """Residual targets in common target-scaled units."""
        residual = data.target - self.prior.predict(data)
        result = np.zeros_like(residual, dtype=np.float32)
        result[:, self.mask] = residual[:, self.mask] / self.scale
        return result.reshape(len(data), -1)

    def forward_scaled(self, output):
        """Apply the physical residual bound in scaled units during training."""
        if self.bounded:
            import tensorflow as tf
            return (self.bound / self.scale) * tf.math.tanh(output)
        return output

    def predict(self, data: PreparedData, gated: bool = False, strength: float = 1.0) -> np.ndarray:
        """Apply a frozen, scenario-independent gate only at inference."""
        raw = self.predict_raw(data)
        correction = self.bound * np.tanh(raw) if self.bounded else self.scale * raw
        if gated:
            if not self.bounded:
                raise ValueError("Registered gated variant requires a bounded residual")
            correction *= gate(data.distance, strength)[:, None, None]
        result = self.prior.predict(data) + correction
        result[:, ~self.mask] = np.nan
        return result
