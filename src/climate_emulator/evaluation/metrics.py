"""Exact ClimateBench averaging order and explicit undefined normalization."""
from __future__ import annotations

import numpy as np
import xarray as xr

from climate_emulator.data.support import area_weights


def common_arrays(reference: xr.DataArray, prediction: xr.DataArray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Require identical coordinate ordering and a single static valid-cell mask."""
    reference, prediction = reference.transpose("year", "lat", "lon"), prediction.transpose("year", "lat", "lon")
    for axis in ("year", "lat", "lon"):
        if not np.array_equal(reference[axis], prediction[axis]):
            raise ValueError(f"Metric coordinate mismatch: {axis}")
    truth, pred = reference.values, prediction.values
    valid = np.isfinite(truth).all(axis=0)
    if mask is not None:
        if mask.shape != valid.shape:
            raise ValueError("Mask shape mismatch")
        valid &= mask
    if not valid.any():
        raise ValueError("No common valid evaluation cells")
    if not np.isfinite(pred[:, valid]).all():
        raise ValueError("Nonfinite model predictions on the common reference mask")
    weights = area_weights(reference.lat.values, reference.sizes["lon"])
    valid &= weights > 0
    if not valid.any():
        raise ValueError("No positive-area evaluation cells")
    w = weights[valid] / weights[valid].sum()
    return truth[:, valid], pred[:, valid], w


def benchmark_metrics(reference: xr.DataArray, prediction: xr.DataArray, mask: np.ndarray | None = None, denominator: float | None = None, tolerance: float = 1e-10, global_weight: float = 5.0) -> dict:
    """Score the supplied period; caller selects 2080–2100 explicitly."""
    truth, pred, w = common_arrays(reference, prediction, mask)
    error = pred - truth
    mean_error = error.mean(axis=0)
    spatial = float(np.sqrt(np.sum(mean_error ** 2 * w)))
    global_error = error @ w
    global_rmse = float(np.sqrt(np.mean(global_error ** 2)))
    bias = float(global_error.mean())
    d = abs(float(truth.mean(axis=0) @ w)) if denominator is None else float(denominator)
    true_map, pred_map = truth.mean(axis=0), pred.mean(axis=0)
    centered_true, centered_pred = true_map - true_map @ w, pred_map - pred_map @ w
    norm = np.sqrt(np.sum(w * centered_true ** 2) * np.sum(w * centered_pred ** 2))
    correlation = float(np.sum(w * centered_true * centered_pred) / norm) if norm > tolerance else None
    defined = bool(np.isfinite(d) and d > tolerance)
    return {"spatial_rmse": spatial, "global_rmse": global_rmse, "bias": bias, "pattern_correlation": correlation, "normalization_denominator": d, "normalization_status": "defined" if defined else "undefined_near_zero_denominator", "nrmse_spatial": spatial / d if defined else None, "nrmse_global": global_rmse / d if defined else None, "nrmse_total": (spatial + global_weight * global_rmse) / d if defined else None, "valid_grid_cells": truth.shape[1], "years": truth.shape[0]}


def annual_rmse(reference: xr.DataArray, prediction: xr.DataArray, mask: np.ndarray | None = None) -> np.ndarray:
    """Area-weighted spatial error for every individual year."""
    truth, pred, w = common_arrays(reference, prediction, mask)
    return np.sqrt(np.sum((pred - truth) ** 2 * w, axis=1))


def period(data: xr.DataArray, years: list[int] | tuple[int, int]) -> xr.DataArray:
    """Select a complete inclusive period, rejecting silently missing years."""
    lower, upper = years
    expected = np.arange(lower, upper + 1)
    result = data.sel(year=slice(lower, upper))
    if not np.array_equal(result.year.values, expected):
        raise ValueError(f"Incomplete evaluation period {lower}–{upper}")
    return result
