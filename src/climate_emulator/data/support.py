"""Emissions envelope distances; no target arrays enter this module."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = ("CO2", "CH4", "SO2", "BC")
BIN_LABELS = ("0", "(0,0.25]", "(0.25,0.5]", "(0.5,1]", ">1")


def area_weights(lat: np.ndarray, nlon: int) -> np.ndarray:
    """Return normalized cosine-latitude weights on a rectangular grid."""
    lat = np.asarray(lat, dtype=float)
    if np.any(np.abs(lat) > 90) or nlon < 1:
        raise ValueError("Invalid latitude/longitude grid")
    weights = np.broadcast_to(np.maximum(np.cos(np.deg2rad(lat)), 0)[:, None], (len(lat), nlon)).copy()
    if weights.sum() <= 0:
        raise ValueError("Grid has no positive area weights")
    return weights / weights.sum()


def support_features(maps: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Area means in CO2, CH4, SO2, BC channel order; units remain native."""
    if maps.ndim != 4 or maps.shape[-1] != 4 or not np.isfinite(maps).all():
        raise ValueError("Expected finite four-channel maps")
    return np.einsum("nlhc,lh->nc", maps, weights)


@dataclass
class SupportEnvelope:
    """Frozen marginal support bounds and explicit constant-feature handling."""
    atol: float = 1e-12
    rtol: float = 1e-10
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "SupportEnvelope":
        """Set extrema on training inputs only."""
        if features.ndim != 2 or features.shape[1] != 4 or not np.isfinite(features).all():
            raise ValueError("Expected finite (sample,4) support features")
        self.minimum, self.maximum = features.min(axis=0), features.max(axis=0)
        return self

    def distances(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return maximum and per-feature dimensionless exceedances."""
        if self.minimum is None or self.maximum is None:
            raise RuntimeError("Support envelope has not been fitted")
        if not np.isfinite(features).all():
            raise ValueError("Support inputs must be finite")
        span = self.maximum - self.minimum
        tolerance = self.atol + self.rtol * np.maximum(np.abs(self.minimum), np.abs(self.maximum))
        constant = span <= tolerance
        raw = np.maximum(np.maximum(self.minimum - features, features - self.maximum), 0)
        raw = np.where(raw <= tolerance, 0, raw)
        per_feature = np.divide(raw, span, out=np.zeros_like(raw, dtype=float), where=~constant)
        per_feature[:, constant] = np.where(raw[:, constant] > 0, np.inf, 0)
        return per_feature.max(axis=1), per_feature

    def summary(self, features: np.ndarray, years: np.ndarray) -> dict:
        """Describe measured violations without inferring from scenario names."""
        distance, each = self.distances(features)
        outside = distance > 0
        bins = distance_bins(distance)
        return {"first_out_of_range_year": int(years[outside][0]) if outside.any() else None, "fraction_outside": float(outside.mean()), "maximum_distance": float(distance.max()), "genuine_marginal_extrapolation": bool(outside.any()), "per_feature_max": dict(zip(FEATURE_NAMES, each.max(axis=0).tolist())), "bin_counts": {name: int(np.sum(bins == i)) for i, name in enumerate(BIN_LABELS)}}


def distance_bins(distance: np.ndarray) -> np.ndarray:
    """Assign fixed bins, preserving zero and including infinite exceedances."""
    distance = np.asarray(distance, dtype=float)
    if np.isnan(distance).any() or (distance < 0).any():
        raise ValueError("Support distances must be nonnegative and not NaN")
    return np.select([distance == 0, distance <= .25, distance <= .5, distance <= 1], [0, 1, 2, 3], default=4)


def gate(distance: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Fixed decreasing gate; positive infinite distance yields zero."""
    if strength <= 0 or np.isnan(distance).any() or (np.asarray(distance) < 0).any():
        raise ValueError("Gate requires positive strength and nonnegative distances")
    return np.exp(-strength * np.asarray(distance, dtype=float))
