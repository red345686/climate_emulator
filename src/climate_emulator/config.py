"""YAML composition and strict experiment configuration validation."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, updates: dict) -> dict:
    """Return a recursive merge; lists are replaced rather than extended."""
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read includes relative to their YAML file and reject cycles."""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"Cyclic configuration include: {path}")
    with path.open(encoding="utf-8") as stream:
        own = yaml.safe_load(stream)
    if not isinstance(own, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    result: dict[str, Any] = {}
    for include in own.pop("include", []):
        result = deep_merge(result, load_config(path.parent / include, (*_seen, path)))
    result = deep_merge(result, own)
    result["_project_root"] = str(path.parent.parent)
    result["_config_path"] = str(path)
    return result


def resolve_path(config: dict, value: str | Path) -> Path:
    """Resolve a project-relative path independently of the process directory."""
    path = Path(value)
    return path if path.is_absolute() else Path(config["_project_root"]) / path


def validate_config(config: dict) -> None:
    """Reject unsupported methods, invalid splits and missing required choices."""
    for section in ("data", "preprocessing", "splits", "models", "training", "evaluation", "statistics", "gate", "experiment"):
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
    d, t, e = config["data"], config["training"], config["experiment"]
    if d["inputs"] != ["CO2", "CH4", "SO2", "BC"]:
        raise ValueError("Input channel order must be CO2, CH4, SO2, BC")
    if d["history"] != 10:
        raise ValueError("The registered protocol requires ten-year histories")
    if not set(d["targets"]) <= {"tas", "pr", "pr90", "diurnal_temperature_range"}:
        raise ValueError("Unsupported target")
    if t["optimizer"] != "rmsprop" or t["protocol"] not in {"legacy", "harmonized"}:
        raise ValueError("Unsupported optimizer or training protocol")
    if t["micro_batch_size"] < 1 or t["effective_batch_size"] < 1 or t["effective_batch_size"] % t["micro_batch_size"]:
        raise ValueError("micro_batch_size must divide effective_batch_size")
    if not t["seeds"] or len(set(t["seeds"])) != len(t["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    if e["mode"] not in {"core", "smoke"}:
        raise ValueError("experiment.mode must be core or smoke")
    allowed = {"lps", "ridge", "rf", "cnn_lstm", "residual_unbounded", "residual_bounded", "gp"}
    if not set(e["models"]) <= allowed:
        raise ValueError("Unknown model in experiment matrix")
    for values, label in ((d["targets"], "targets"), (e["models"], "models"), (e["splits"], "splits")):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"Configured {label} must be nonempty and unique")
    for key in ("learning_rate", "gradient_clipnorm", "max_epochs", "patience", "fit_budget_seconds", "project_budget_seconds", "inference_batch_size"):
        if t[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if t["kernel_l2"] < 0 or t["min_delta"] < 0:
        raise ValueError("Regularization and minimum improvement must be nonnegative")
    if config["preprocessing"]["input_scaling"] != "channel_standardization":
        raise ValueError("Only registered channel standardization is implemented")
    for key in ("historical_years", "future_years"):
        if len(d[key]) != 2 or d[key][0] > d[key][1]:
            raise ValueError(f"Invalid data.{key}")
    lower, upper = config["evaluation"]["benchmark_years"]
    if not d["future_years"][0] <= lower <= upper <= d["future_years"][1]:
        raise ValueError("Benchmark years must be an inclusive subset of future years")
    if "gp" in e["models"] and not config["models"]["gp"]["enabled"]:
        raise ValueError("GP explicitly disabled")
    if config["preprocessing"]["target_scaling"] != "rms":
        raise ValueError("Only registered scalar RMS target scaling is implemented")
    for name in e["splits"]:
        s = config["splits"][name]
        if len(set(s["train"])) != len(s["train"]) or len(set(s["test"])) != len(s["test"]):
            raise ValueError(f"Duplicate scenario in split {name}")
        if s["validation"] not in s["train"] or set(s["train"]) & set(s["test"]):
            raise ValueError(f"Invalid training/validation/test split: {name}")
        if "historical" not in s["train"] or not set(s["train"] + s["test"]) <= set(d["required_scenarios"]):
            raise ValueError(f"Unregistered scenario or missing historical context: {name}")
