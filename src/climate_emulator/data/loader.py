"""Coordinate-aware processed ClimateBench loading; no member averaging."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


@dataclass
class ScenarioData:
    """Loaded input/target pair and original file provenance."""
    name: str
    inputs: xr.Dataset
    targets: xr.Dataset
    input_path: Path
    target_path: Path


def discover(root: str | Path) -> dict[str, dict[str, Path]]:
    """Discover conventionally named files and reject ambiguous duplicates."""
    found: dict[str, dict[str, Path]] = {}
    for path in sorted(Path(root).rglob("*.nc")):
        for prefix, kind in (("inputs_", "inputs"), ("outputs_", "targets")):
            if path.stem.startswith(prefix):
                scenario = path.stem[len(prefix):]
                pair = found.setdefault(scenario, {})
                if kind in pair:
                    raise ValueError(f"Duplicate {kind} for {scenario}: {pair[kind]} and {path}")
                pair[kind] = path
    return found


def canonicalize(ds: xr.Dataset) -> xr.Dataset:
    """Rename coordinate aliases and preserve explicit, ordered calendar years."""
    rename = {old: new for old, new in (("latitude", "lat"), ("longitude", "lon"), ("time", "year")) if old in ds.dims}
    for old, new in rename.items():
        if new in ds.dims or new in ds.coords:
            raise ValueError(f"Ambiguous coordinate aliases: {old}/{new}")
    ds = ds.rename(rename)
    if "year" not in ds.coords:
        raise ValueError("No explicit year/time coordinate")
    coord = ds["year"]
    if np.issubdtype(coord.dtype, np.number):
        years = coord.values
        if not np.isfinite(years).all() or not np.equal(years, np.round(years)).all():
            raise ValueError("Numeric time must contain integer calendar years")
        years = years.astype(int)
    else:
        years = coord.dt.year.values.astype(int)
    if len(years) < 1 or np.any(np.diff(years) <= 0):
        raise ValueError("Years must be unique and strictly increasing")
    ds = ds.assign_coords(year=years)
    for axis in ("lat", "lon", "member"):
        if axis in ds.dims:
            if axis not in ds.coords or len(np.unique(ds[axis].values)) != ds.sizes[axis]:
                raise ValueError(f"Missing or duplicate coordinate: {axis}")
    if "quantile" in ds.dims and ds.sizes["quantile"] == 1:
        ds = ds.squeeze("quantile", drop=True)
    return ds


def load_dataset(path: str | Path) -> xr.Dataset:
    """Load into owned memory, then close the NetCDF handle."""
    with xr.open_dataset(path) as ds:
        return canonicalize(ds).load()


def assert_grid_equal(left: xr.Dataset | xr.DataArray, right: xr.Dataset | xr.DataArray) -> None:
    """Reject even reordered grids; never align by array position."""
    for axis in ("lat", "lon"):
        if axis not in left.coords or axis not in right.coords or not np.array_equal(left[axis], right[axis]):
            raise ValueError(f"Grid mismatch on {axis}")


def load_scenario(root: str | Path, scenario: str) -> ScenarioData:
    """Load a complete pair and validate years/grid/member coordinates."""
    pair = discover(root).get(scenario, {})
    if set(pair) != {"inputs", "targets"}:
        raise FileNotFoundError(f"Missing input/target pair for {scenario}: {pair}")
    inputs, targets = load_dataset(pair["inputs"]), load_dataset(pair["targets"])
    assert_grid_equal(inputs, targets)
    if not np.array_equal(inputs.year, targets.year):
        raise ValueError(f"Input/target year mismatch: {scenario}")
    if "member" not in targets.dims or targets.sizes["member"] < 1:
        raise ValueError(f"No ensemble member dimension: {scenario}")
    return ScenarioData(scenario, inputs, targets, pair["inputs"], pair["targets"])


def convert_targets(ds: xr.Dataset, assumptions: dict[str, str]) -> xr.Dataset:
    """Convert explicitly recognized units and record every assumption/conversion."""
    result = ds.copy(deep=True)
    for name in result.data_vars:
        if name not in {"tas", "pr", "pr90", "diurnal_temperature_range"}:
            continue
        original = result[name].attrs.get("units")
        unit = original or assumptions.get(name)
        if unit is None:
            raise ValueError(f"Units missing for {name}; configure an explicit assumption")
        normalized = str(unit).strip().lower().replace("**", "").replace("^", "")
        factor = 1.0
        if name in {"pr", "pr90"}:
            if normalized in {"kg m-2 s-1", "kg/m2/s", "kg m−2 s−1"}:
                factor = 86400.0
            elif normalized not in {"mm/day", "mm day-1", "mm d-1"}:
                raise ValueError(f"Unsupported precipitation units: {unit}")
            output_unit = "mm/day"
        else:
            if normalized not in {"k", "kelvin", "degc", "degrees_celsius", "c", "°c"}:
                raise ValueError(f"Unsupported temperature anomaly units: {unit}")
            output_unit = "K"
        attrs = dict(result[name].attrs)
        result[name] = result[name] * factor
        result[name].attrs = {**attrs, "units": output_unit, "source_units": str(unit), "unit_assumed": int(original is None), "conversion_factor": factor, "quantity": "anomaly"}
    return result


def input_maps(ds: xr.Dataset, names: list[str]) -> np.ndarray:
    """Broadcast scalar GHG inputs to the native grid in documented channel order."""
    if "lat" not in ds.dims or "lon" not in ds.dims:
        raise ValueError("Spatial input coordinates missing")
    template = xr.DataArray(np.empty((ds.sizes["year"], ds.sizes["lat"], ds.sizes["lon"])), dims=("year", "lat", "lon"), coords={k: ds[k] for k in ("year", "lat", "lon")})
    channels = []
    for name in names:
        if name not in ds:
            raise ValueError(f"Required input missing: {name}")
        if not set(ds[name].dims) <= {"year", "lat", "lon"}:
            raise ValueError(f"Unexpected input dimensions for {name}: {ds[name].dims}")
        array = ds[name].broadcast_like(template).transpose("year", "lat", "lon").values
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite required inputs: {name}")
        if name in {"CO2", "CH4"} and not np.allclose(array, array[:, :1, :1], rtol=1e-10, atol=0):
            raise ValueError(f"Global scalar input is spatially nonuniform: {name}")
        channels.append(array)
    return np.stack(channels, axis=-1)
