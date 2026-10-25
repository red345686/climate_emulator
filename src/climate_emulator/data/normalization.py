"""Serializable training-only input normalization and scalar target scaling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelNormalizer:
    """Fit one mean/std per input channel, with constant-channel protection."""
    floor: float = 1e-8
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, inputs: np.ndarray) -> "ChannelNormalizer":
        """Fit on unique training input maps only."""
        if not np.isfinite(inputs).all() or inputs.ndim != 4:
            raise ValueError("Expected finite (year,lat,lon,channel) inputs")
        self.mean = inputs.mean(axis=(0, 1, 2), dtype=np.float64)
        std = inputs.std(axis=(0, 1, 2), dtype=np.float64)
        # Aerosol fluxes can be far smaller than 1e-8 in native units. An
        # absolute target-scale floor must not erase their input variation.
        self.scale = np.where(std > 0, std, 1.0)
        return self

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        """Apply frozen parameters; never fit or clip evaluation inputs."""
        if self.mean is None or self.scale is None:
            raise RuntimeError("Input normalizer has not been fitted")
        if not np.isfinite(inputs).all():
            raise ValueError("Nonfinite required inputs")
        return ((inputs - self.mean) / self.scale).astype(np.float32)


@dataclass
class TargetScaler:
    """Single area-weighted physical RMS per target, without centering."""
    floor: float = 1e-8
    scale: float | None = None

    def fit(self, target: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> "TargetScaler":
        """Fit one scale over training endpoints and valid cells."""
        selected = target[:, mask]
        if not np.isfinite(selected).all():
            raise ValueError("Nonfinite targets within training mask")
        w = weights[mask] / weights[mask].sum()
        self.scale = max(float(np.sqrt(np.mean(np.sum(selected ** 2 * w, axis=1)))), self.floor)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Convert physical anomalies to scaled numeric units."""
        if self.scale is None:
            raise RuntimeError("Target scaler has not been fitted")
        return values / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Restore physical units."""
        if self.scale is None:
            raise RuntimeError("Target scaler has not been fitted")
        return values * self.scale
