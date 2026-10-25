"""Small deterministic fixtures; no ClimateBench download is required."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from climate_emulator.config import load_config
from climate_emulator.data.preprocessing import Preprocessor, collect_samples


@pytest.fixture
def config(tmp_path: Path) -> dict:
    """Isolated synthetic project configuration."""
    project = Path(__file__).resolve().parents[1]
    value = load_config(project / "configs" / "core.yaml")
    value["_project_root"] = str(tmp_path)
    value["data"]["root"] = "raw"
    value["data"]["require_archives"] = False
    value["data"]["expected_grid"] = [4, 6]
    value["training"]["max_epochs"] = 2
    value["training"]["patience"] = 2
    value["training"]["micro_batch_size"] = 4
    value["training"]["effective_batch_size"] = 8
    value["models"]["rf"].update(n_estimators=5, max_depth=[3], min_samples_leaf=[2])
    return value


@pytest.fixture
def dataset(config: dict) -> Path:
    """Write tiny synthetic analogues with complete registered year coverage."""
    root = Path(config["_project_root"]) / "raw"
    root.mkdir()
    lat, lon = np.array([-67.5, -22.5, 22.5, 67.5]), np.arange(6) * 60
    pattern = 1 + np.sin(np.deg2rad(lat))[:, None] * np.ones((1, 6)) * .2
    for index, scenario in enumerate(config["data"]["required_scenarios"]):
        years = np.arange(2015, 2101) if scenario.startswith("ssp") else np.arange(1850, 2015)
        driver = (years - 1850) * (1 + .1 * index)
        fields = driver[:, None, None] * pattern[None] * .001
        inp = xr.Dataset({"CO2": ("time", driver), "CH4": ("time", driver * .001), "SO2": (("time", "latitude", "longitude"), fields), "BC": (("time", "latitude", "longitude"), fields * .3)}, coords={"time": years, "latitude": lat, "longitude": lon})
        tas = .01 * driver[:, None, None] * pattern[None]
        member_fields = np.stack([tas - .01, tas, tas + .01])
        out = xr.Dataset({"tas": (("member", "time", "lat", "lon"), member_fields), "pr": (("member", "time", "lat", "lon"), member_fields * .05 / 86400)}, coords={"member": [1, 2, 3], "time": years, "lat": lat, "lon": lon})
        out.tas.attrs["units"] = "K"
        out.pr.attrs["units"] = "kg m-2 s-1"
        inp.to_netcdf(root / f"inputs_{scenario}.nc")
        out.to_netcdf(root / f"outputs_{scenario}.nc")
    return root


@pytest.fixture
def prepared(dataset: Path, config: dict):
    """Training-only preprocessing on small historical samples."""
    raw = collect_samples(dataset, ["historical"], "tas", config, limit=20)
    pipeline = Preprocessor(config).fit(raw)
    return pipeline.transform(raw), pipeline, raw
