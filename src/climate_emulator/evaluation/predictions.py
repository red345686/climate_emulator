"""Coordinate-bearing NetCDF prediction serialization with provenance."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import xarray as xr

from climate_emulator.data.preprocessing import PreparedData
from climate_emulator.utils import clean_json


def prediction_array(values: np.ndarray, data: PreparedData, scenario: str) -> xr.DataArray:
    """Extract one scenario with exact endpoint years and spatial coordinates."""
    positions = [i for i, key in enumerate(data.keys) if key[0] == scenario]
    return xr.DataArray(values[positions], dims=("year", "lat", "lon"), coords={"year": [data.keys[i][1] for i in positions], "lat": data.lat, "lon": data.lon}, name=data.target_name)


def save_predictions(path: Path, values: np.ndarray, data: PreparedData, scenario: str, provenance: dict) -> None:
    """Atomically save one scenario/target/seed without positional alignment."""
    array = prediction_array(values, data, scenario).expand_dims(scenario=[scenario])
    ds = array.to_dataset()
    indices = [i for i, key in enumerate(data.keys) if key[0] == scenario]
    ds["support_distance"] = (("scenario", "year"), data.distance[indices][None, :])
    ds["training_mask"] = (("lat", "lon"), data.mask.astype(np.int8))
    ds[data.target_name].attrs["units"] = "mm/day" if data.target_name in {"pr", "pr90"} else "K"
    ds[data.target_name].attrs["quantity"] = "anomaly"
    ds.attrs["provenance_json"] = json.dumps(clean_json(provenance), sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".{os.getpid()}.tmp.nc")
    ds.to_netcdf(temporary)
    temporary.replace(path)


def load_predictions(path: Path) -> tuple[xr.DataArray, dict, np.ndarray, np.ndarray]:
    """Read predictions into owned memory and close their file handle."""
    with xr.open_dataset(path) as source:
        ds = source.load()
    provenance = json.loads(ds.attrs["provenance_json"])
    target = provenance["target"]
    if ds.sizes["scenario"] != 1:
        raise ValueError("Expected one scenario per prediction artifact")
    return ds[target].isel(scenario=0, drop=True), provenance, ds.training_mask.values.astype(bool), ds.support_distance.isel(scenario=0).values
