"""Member-preserving stage manifests and explicit target/context isolation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from climate_emulator.data.loader import load_scenario


def endpoint_years(scenario: str, years: np.ndarray, data_config: dict) -> np.ndarray:
    """Select registered endpoints without changing a split on the basis of errors."""
    lower, upper = data_config["future_years"] if scenario.startswith("ssp") else data_config["historical_years"]
    return years[(years >= lower) & (years <= upper)]


def build_manifest(root: Path, split: dict, data_config: dict, stage: str) -> pd.DataFrame:
    """Describe every input year and target member for selection or final fitting."""
    if stage not in {"selection", "final"}:
        raise ValueError("Manifest stage must be selection or final")
    partitions = {s: "train" for s in split["train"]}
    if stage == "selection":
        partitions[split["validation"]] = "validation"
    partitions.update({s: "test" for s in split["test"]})
    records = []
    for scenario, partition in partitions.items():
        pair = load_scenario(root, scenario)
        endpoints = set(endpoint_years(scenario, pair.inputs.year.values, data_config).tolist())
        for year in pair.inputs.year.values:
            records.append({"scenario": scenario, "year": int(year), "member": "input", "partition": partition, "role": "input", "valid": True, "exclusion_reason": ""})
            for member in pair.targets.member.values:
                eligible = int(year) in endpoints
                any_values = all(bool(np.isfinite(pair.targets[v].sel(year=year, member=member)).any()) for v in data_config["targets"])
                records.append({"scenario": scenario, "year": int(year), "member": str(member), "partition": partition, "role": "target" if eligible and any_values else "excluded_target", "valid": bool(eligible and any_values), "exclusion_reason": "" if eligible and any_values else "outside_common_endpoint_range" if not eligible else "entire_member_field_nonfinite"})
    # SSP history explicitly refers to canonical historical inputs, never outputs.
    for scenario, partition in partitions.items():
        if scenario.startswith("ssp"):
            first = data_config["future_years"][0]
            for year in range(first - data_config["history"] + 1, first):
                records.append({"scenario": "historical", "year": year, "member": "input", "partition": partition, "role": "input_context", "valid": True, "exclusion_reason": "", "context_for": scenario})
    frame = pd.DataFrame(records).fillna({"context_for": ""})
    validate_manifest(frame, split, stage, data_config)
    return frame


def validate_manifest(frame: pd.DataFrame, split: dict, stage: str, data_config: dict | None = None) -> None:
    """Reject duplicate targets, overlapping scenarios and invalid target partitions."""
    target = frame[frame.role == "target"]
    if target.duplicated(["scenario", "year", "member"]).any():
        raise ValueError("A scenario/year/member target occurs more than once")
    if target.groupby("scenario").partition.nunique().gt(1).any():
        raise ValueError("Scenario targets cross partitions")
    train = set(target.loc[target.partition == "train", "scenario"])
    expected = set(split["train"]) - ({split["validation"]} if stage == "selection" else set())
    if train != expected:
        raise ValueError(f"Training scenarios differ from registered pool: {train} != {expected}")
    if set(target.loc[target.partition == "test", "scenario"]) != set(split["test"]):
        raise ValueError("Outer-test manifest changed")
    expected_validation = {split["validation"]} if stage == "selection" else set()
    if set(target.loc[target.partition == "validation", "scenario"]) != expected_validation:
        raise ValueError("Inner-validation manifest changed")
    if not target.valid.all():
        raise ValueError("Entire target member fields are missing; audit before proceeding")
    if not set(frame.partition) <= {"train", "validation", "test"}:
        raise ValueError("Unknown manifest partition")
    if data_config is not None:
        for scenario, group in target.groupby("scenario"):
            bounds = data_config["future_years"] if scenario.startswith("ssp") else data_config["historical_years"]
            expected_years = set(range(bounds[0], bounds[1] + 1))
            if set(group.year) != expected_years:
                raise ValueError(f"Manifest endpoint coverage changed: {scenario}")
            # Member counts may change by year in the verified processed release.
            # Complete endpoint coverage above still requires a member every year.
    for partition in ("train", "validation", "test"):
        historical = target[(target.scenario == "historical") & (target.partition == partition)]
        if historical.duplicated(["year", "member"]).any():
            raise ValueError("Duplicated historical targets")


def write_manifests(root: Path, destination: Path, splits: dict, data_config: dict) -> None:
    """Write both stages; execution is only initiated by an explicit CLI."""
    destination.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        for stage in ("selection", "final"):
            build_manifest(root, split, data_config, stage).to_csv(destination / f"{name}_{stage}.csv", index=False)
