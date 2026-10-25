"""Training-only PCA and a fitted preprocessing object shared by all models."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from climate_emulator.data.loader import assert_grid_equal, convert_targets, input_maps, load_scenario
from climate_emulator.data.manifests import endpoint_years
from climate_emulator.data.members import available_member_mean
from climate_emulator.data.normalization import ChannelNormalizer, TargetScaler
from climate_emulator.data.support import SupportEnvelope, area_weights, support_features
from climate_emulator.data.windows import HistoryStore, window_indices


class WeightedPCA:
    """Centered area-weighted PCA with a cap and recorded achieved variance."""

    def __init__(self, components: int, variance: float | None = None, seed: int = 17):
        self.components, self.variance, self.seed = components, variance, seed

    def fit(self, fields: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> "WeightedPCA":
        """Fit on finite training cells and samples only."""
        self.shape, self.mask = fields.shape[1:], mask.copy()
        self.sqrt_weights = np.sqrt(weights[mask] / weights[mask].sum())
        matrix = fields[:, mask] * self.sqrt_weights
        if not np.isfinite(matrix).all() or len(matrix) < 2:
            raise ValueError("PCA requires at least two finite training samples")
        count = min(self.components, len(matrix) - 1, matrix.shape[1])
        if count < 1:
            raise ValueError("No valid PCA components")
        self.mean = matrix.mean(axis=0)
        self.constant = bool(np.all(np.ptp(matrix, axis=0) == 0))
        if self.constant:
            self.pca = None
            self.retained = 1
            self.achieved_variance = 1.0
        else:
            self.pca = PCA(n_components=count, svd_solver="randomized", random_state=self.seed).fit(matrix)
            cumulative = np.cumsum(self.pca.explained_variance_ratio_)
            self.retained = min(int(np.searchsorted(cumulative, self.variance)) + 1, count) if self.variance else count
            self.achieved_variance = float(cumulative[self.retained - 1])
        return self

    def transform(self, fields: np.ndarray) -> np.ndarray:
        """Project using the frozen training basis."""
        matrix = fields[:, self.mask] * self.sqrt_weights
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite field within fitted PCA mask")
        if self.pca is None:
            return np.zeros((len(fields), self.retained))
        return self.pca.transform(matrix)[:, :self.retained]

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        """Restore physical fields, leaving excluded grid cells undefined."""
        matrix = np.broadcast_to(self.mean, (len(scores), len(self.mean))).copy()
        if self.pca is not None:
            matrix += scores @ self.pca.components_[:self.retained]
        result = np.full((len(scores), *self.shape), np.nan)
        result[:, self.mask] = matrix / self.sqrt_weights
        return result

    def reconstruction_rmse(self, fields: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return dimensional weighted reconstruction error per sample."""
        reconstructed = self.inverse_transform(self.transform(fields))
        w = weights[self.mask] / weights[self.mask].sum()
        return np.sqrt(np.sum((reconstructed[:, self.mask] - fields[:, self.mask]) ** 2 * w, axis=1))


class AerosolFeatures:
    """Current GHG values and padded training-only spatial aerosol PC scores."""

    def __init__(self, components: int = 5):
        self.components = components

    def fit(self, maps: np.ndarray, weights: np.ndarray) -> "AerosolFeatures":
        """Fit independent area-weighted BC/SO2 bases."""
        mask = np.ones(weights.shape, dtype=bool)
        self.weights = weights.copy()
        self.bases = {channel: WeightedPCA(self.components).fit(maps[..., channel], weights, mask) for channel in (3, 2)}
        return self

    def transform(self, maps: np.ndarray) -> np.ndarray:
        """Keep a fixed 12-feature interface even for small smoke grids."""
        ghg = support_features(maps, self.weights)[:, :2]
        pieces = [ghg]
        for channel in (3, 2):
            scores = self.bases[channel].transform(maps[..., channel])
            pieces.append(np.pad(scores, ((0, 0), (0, self.components - scores.shape[1]))))
        return np.concatenate(pieces, axis=1)


@dataclass
class RawSamples:
    """Physical training/evaluation endpoints and unique raw input histories."""
    maps: np.ndarray
    indices: np.ndarray
    target: np.ndarray
    temperature: np.ndarray
    keys: list[tuple[str, int]]
    input_keys: list[tuple[str, int]]
    lat: np.ndarray
    lon: np.ndarray
    target_name: str
    members: dict[str, list]

    @property
    def current_maps(self) -> np.ndarray:
        """Physical input maps at each target endpoint."""
        return self.maps[self.indices[:, -1]]


@dataclass
class PreparedData:
    """Frozen transformed inputs with physical targets and endpoint provenance."""
    history: HistoryStore
    features: np.ndarray
    support: np.ndarray
    distance: np.ndarray
    target: np.ndarray
    temperature: np.ndarray
    keys: list[tuple[str, int]]
    lat: np.ndarray
    lon: np.ndarray
    target_name: str
    mask: np.ndarray
    weights: np.ndarray
    scale: float
    diagnostics: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.keys)


def collect_samples(root: Path, scenarios: list[str], target: str, config: dict, limit: int | None = None, spatial_slice: tuple[int | None, int | None] | None = None) -> RawSamples:
    """Collect only requested scenario targets; historical inputs are reusable context."""
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("Duplicate scenario requested")
    d = config["data"]
    pairs = {s: load_scenario(root, s) for s in set(scenarios) | {"historical"}}
    grid = pairs["historical"].inputs
    for pair in pairs.values():
        assert_grid_equal(grid, pair.inputs)
    if spatial_slice:
        nlat, nlon = spatial_slice
        for pair in pairs.values():
            selection = {"lat": slice(0, nlat), "lon": slice(0, nlon)}
            pair.inputs = pair.inputs.isel(**selection)
            pair.targets = pair.targets.isel(**selection)
    unique_maps, input_keys, mapping, indices, targets, temperatures, keys = [], [], {}, [], [], [], []
    members = {}
    for scenario in scenarios:
        pair = pairs[scenario]
        output = convert_targets(pair.targets, d["missing_unit_assumptions"])
        if target not in output or "tas" not in output:
            raise ValueError(f"Missing {target}/tas in {scenario}")
        years = endpoint_years(scenario, pair.inputs.year.values, d)
        if limit is not None:
            years = years[:max(limit - len(keys), 0)]
        if not len(years):
            continue
        if scenario.startswith("ssp"):
            hist = pairs["historical"].inputs.sel(year=slice(int(years.min()) - d["history"] + 1, d["future_years"][0] - 1))
            input_ds = xr.concat([hist, pair.inputs], dim="year", data_vars="all")
            sources = ["historical"] * hist.sizes["year"] + [scenario] * pair.inputs.sizes["year"]
        else:
            input_ds = pair.inputs
            sources = [scenario] * input_ds.sizes["year"]
        windows = window_indices(input_ds.year.values, years, d["history"])
        required = np.unique(windows)
        values = input_maps(input_ds.isel(year=required), d["inputs"])
        position_lookup = {}
        for local, source_index in enumerate(required):
            key = (sources[source_index], int(input_ds.year.values[source_index]))
            if key not in mapping:
                mapping[key] = len(unique_maps)
                unique_maps.append(values[local])
                input_keys.append(key)
            position_lookup[int(source_index)] = mapping[key]
        indices.extend([[position_lookup[int(i)] for i in row] for row in windows])
        # Whole absent member-years are excluded; missing cells in present fields
        # still propagate into the static mask rather than being imputed.
        targets.extend(available_member_mean(output[target].sel(year=years)).transpose("year", "lat", "lon").values)
        temperatures.extend(available_member_mean(output.tas.sel(year=years)).transpose("year", "lat", "lon").values)
        keys.extend((scenario, int(year)) for year in years)
        members[scenario] = output.member.values.tolist()
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("No endpoints or duplicated scenario-year targets")
    first = pairs["historical"].inputs
    return RawSamples(np.asarray(unique_maps), np.asarray(indices), np.asarray(targets), np.asarray(temperatures), keys, input_keys, first.lat.values, first.lon.values, target, members)


class Preprocessor:
    """One fitted pipeline per split, target, and selection/refit stage."""

    def __init__(self, config: dict):
        self.config = config

    def fit(self, raw: RawSamples) -> "Preprocessor":
        """Fit all transformations from a training-only RawSamples object."""
        p = self.config["preprocessing"]
        self.lat, self.lon = raw.lat.copy(), raw.lon.copy()
        self.keys = list(raw.keys)
        self.input_keys = list(raw.input_keys)
        self.weights = area_weights(raw.lat, len(raw.lon))
        self.mask = np.isfinite(raw.target).all(axis=0) & (self.weights > 0)
        if not self.mask.any():
            raise ValueError("No consistently finite training target cells")
        self.normalizer = ChannelNormalizer(p["scale_floor"]).fit(raw.maps)
        self.scaler = TargetScaler(p["scale_floor"]).fit(raw.target, self.weights, self.mask)
        self.aerosols = AerosolFeatures(p["aerosol_components"]).fit(raw.current_maps, self.weights)
        support = support_features(raw.maps, self.weights)
        self.envelope = SupportEnvelope(p["support_atol"], p["support_rtol"]).fit(support)
        history_vectors = support[raw.indices].reshape(len(raw.keys), -1)
        self.history_scaler = StandardScaler().fit(history_vectors)
        self.neighbors = NearestNeighbors(n_neighbors=1).fit(self.history_scaler.transform(history_vectors))
        return self

    def transform(self, raw: RawSamples) -> PreparedData:
        """Apply immutable training state to training, validation or test data."""
        if not np.array_equal(raw.lat, self.lat) or not np.array_equal(raw.lon, self.lon):
            raise ValueError("Evaluation grid differs from fitted grid")
        support = support_features(raw.current_maps, self.weights)
        distance, per_feature = self.envelope.distances(support)
        all_support = support_features(raw.maps, self.weights)
        history_vectors = all_support[raw.indices].reshape(len(raw.keys), -1)
        diagnostics = {
            "per_feature_distance": per_feature,
            "history_nearest_distance": self.neighbors.kneighbors(self.history_scaler.transform(history_vectors), return_distance=True)[0][:, 0],
            "aerosol_reconstruction": {str(channel): self.aerosols.bases[channel].reconstruction_rmse(raw.current_maps[..., channel], self.weights) for channel in (2, 3)},
        }
        return PreparedData(HistoryStore(self.normalizer.transform(raw.maps), raw.indices), self.aerosols.transform(raw.current_maps), support, distance, raw.target, raw.temperature, raw.keys, raw.lat, raw.lon, raw.target_name, self.mask.copy(), self.weights.copy(), float(self.scaler.scale), diagnostics)

    def summary(self) -> dict:
        """Export fitting provenance and scalar parameters for review."""
        return {"training_target_keys": self.keys, "training_input_keys": self.input_keys, "input_mean": self.normalizer.mean, "input_scale": self.normalizer.scale, "target_scale": self.scaler.scale, "valid_grid_cells": int(self.mask.sum()), "support_minimum": self.envelope.minimum, "support_maximum": self.envelope.maximum, "aerosol_components": {str(k): v.retained for k, v in self.aerosols.bases.items()}}
