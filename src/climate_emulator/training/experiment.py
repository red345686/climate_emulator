"""Leakage-isolated selection, protocol freezing, refitting and resume logic."""
from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import time
import traceback
import platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from climate_emulator.config import resolve_path, validate_config
from climate_emulator.data.loader import discover
from climate_emulator.data.manifests import validate_manifest
from climate_emulator.data.preprocessing import Preprocessor, collect_samples
from climate_emulator.evaluation.metrics import benchmark_metrics, period
from climate_emulator.evaluation.predictions import prediction_array, save_predictions
from climate_emulator.models.base import Emulator
from climate_emulator.models.pattern_scaling import PatternScaling
from climate_emulator.models.ridge import RidgeEmulator
from climate_emulator.models.random_forest import RandomForestEmulator
from climate_emulator.statistics.comparisons import select_incumbent
from climate_emulator.training.callbacks import FitBudgetExceeded
from climate_emulator.utils import file_hash, object_hash, source_hash, read_json, write_json

LOGGER = logging.getLogger(__name__)
NEURAL = {"cnn_lstm", "residual_unbounded", "residual_bounded"}


def environment_versions() -> dict:
    """Record installed versions without importing optional compute backends."""
    versions = {}
    for name in ("numpy", "scipy", "pandas", "xarray", "dask", "netCDF4", "cftime", "scikit-learn", "matplotlib", "PyYAML", "joblib", "tensorflow", "keras", "gpflow", "tensorflow-probability"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def protocol_configuration(config: dict) -> dict:
    """Fingerprint scientific settings without unrelated paper-output choices."""
    return {k: v for k, v in config.items() if not k.startswith("_") and k not in {"paper"}}


def preflight(config: dict) -> dict:
    """Require unchanged audited data, valid manifests and input-only support audit."""
    validate_config(config)
    destination = resolve_path(config, config["data"]["manifests"])
    report = read_json(destination / "dataset_audit.json")
    if not report["complete"] or report["release"] != "7064308":
        raise ValueError("A complete ClimateBench v1 integrity audit is required")
    if config["data"]["require_archives"] and len(report.get("archives", [])) != 2:
        raise ValueError("Audit must verify both required release archives")
    discovered = discover(resolve_path(config, config["data"]["root"]))
    recorded = {Path(item["file"]).resolve(): item for item in report["files"]}
    for scenario in config["data"]["required_scenarios"]:
        for path in discovered.get(scenario, {}).values():
            if path.resolve() not in recorded or file_hash(path) != recorded[path.resolve()]["sha256"]:
                raise ValueError(f"Unchanged audited data required: {path}")
        if set(discovered.get(scenario, {})) != {"inputs", "targets"}:
            raise FileNotFoundError(f"Missing required scenario: {scenario}")
    if not (destination / "support_summary.json").is_file():
        raise FileNotFoundError("Run the input-only support audit before training")
    provenance = read_json(destination / "support_provenance.json")
    for field in ("inputs", "history", "historical_years", "future_years"):
        if provenance[field] != config["data"][field]:
            raise ValueError(f"Support audit configuration changed: {field}")
    if provenance["preprocessing"] != config["preprocessing"]:
        raise ValueError("Support audit preprocessing configuration changed")
    for scenario in config["data"]["required_scenarios"]:
        if provenance["input_hashes"][scenario] != file_hash(discovered[scenario]["inputs"]):
            raise ValueError(f"Support audit input changed: {scenario}")
    support_hashes = {"support_summary.json": file_hash(destination / "support_summary.json"), "support_provenance.json": file_hash(destination / "support_provenance.json")}
    manifest_hashes = {}
    for split in config["experiment"]["splits"]:
        if provenance["splits"].get(split) != config["splits"][split]:
            raise ValueError(f"Support audit split changed: {split}")
        for stage in ("selection", "final"):
            for scenario in dict.fromkeys(config["splits"][split]["train"] + config["splits"][split]["test"]):
                support_path = destination / f"support_{split}_{stage}_{scenario}.csv"
                support_hashes[support_path.name] = file_hash(support_path)
            path = destination / f"{split}_{stage}.csv"
            frame = pd.read_csv(path, keep_default_na=False)
            validate_manifest(frame, config["splits"][split], stage, config["data"])
            for pair in report["pairs"]:
                rows = frame[(frame.scenario == pair["scenario"]) & (frame.role == "target")]
                if not rows.empty and set(rows.member.astype(str)) != set(map(str, pair["members"])):
                    raise ValueError(f"Manifest members differ from audited data: {pair['scenario']}")
            manifest_hashes[path.name] = file_hash(path)
    return {"configuration_hash": object_hash(protocol_configuration(config)), "source_hash": source_hash(Path(config["_project_root"])), "dataset_hash": report["dataset_hash"], "manifest_hashes": manifest_hashes, "support_hashes": support_hashes}


def model_candidates(name: str, config: dict) -> list[dict]:
    """Return only the registered hyperparameter search, never test-driven options."""
    if name == "ridge":
        return [{"alpha": alpha} for alpha in config["models"]["ridge"]["alphas"]]
    if name == "rf":
        r = config["models"]["rf"]
        return [{"n_estimators": r["n_estimators"], "max_depth": depth, "min_samples_leaf": leaf, "max_features": r["max_features"], "n_jobs": r["n_jobs"]} for depth, leaf in itertools.product(r["max_depth"], r["min_samples_leaf"])]
    return [{}]


def make_model(name: str, config: dict, seed: int, parameters: dict, epochs: int | None, log_directory: Path):
    """Instantiate only the selected implementation and its required backend."""
    if name == "lps":
        return PatternScaling()
    if name == "ridge":
        return RidgeEmulator(**parameters)
    if name == "rf":
        p = config["preprocessing"]
        return RandomForestEmulator(parameters, p["target_components"], p["target_variance"], seed)
    if name == "gp":
        from climate_emulator.models.gaussian_process import GaussianProcessEmulator
        return GaussianProcessEmulator(config["models"]["gp"]["maxiter"])
    if name == "cnn_lstm":
        from climate_emulator.models.cnn_lstm import CNNLSTMEmulator
        return CNNLSTMEmulator(config, seed, epochs, log_directory)
    if name in {"residual_bounded", "residual_unbounded"}:
        from climate_emulator.models.residual import ResidualEmulator
        return ResidualEmulator(config, seed, name == "residual_bounded", epochs, log_directory)
    raise ValueError(f"Unknown model: {name}")


def _fit_worker(payload_path: str, destination: str, queue) -> None:
    """Child-process entry point permitting hard RF/GP wall-clock deadlines."""
    try:
        model, data = joblib.load(payload_path)
        model.fit(data)
        model.save(destination)
        queue.put({"status": "complete"})
    except Exception as exc:
        # Deliberately catch at a process boundary; propagate with original details.
        queue.put({"status": "failed", "exception": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise


def fit_with_deadline(model, data, validation, name: str, config: dict, directory: Path, remaining_seconds: float | None = None):
    """Fit RF/GP in an isolated process and neural models with batch deadlines."""
    if name not in {"rf", "gp"}:
        return model.fit(data, validation)
    seconds = remaining_seconds if remaining_seconds is not None else config["models"][name]["budget_seconds"]
    if seconds <= 0:
        raise FitBudgetExceeded(f"No remaining {name} fit budget")
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="fit-", dir=directory) as temporary:
        temporary = Path(temporary)
        payload = temporary / "payload.joblib"
        joblib.dump((model, data), payload)
        context = mp.get_context("spawn")
        queue = context.Queue()
        process = context.Process(target=_fit_worker, args=(str(payload), str(temporary / "checkpoint"), queue))
        process.start()
        process.join(seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            queue.close()
            raise FitBudgetExceeded(f"{name} exceeded remaining budget of {seconds:.1f} seconds")
        if process.exitcode != 0:
            details = queue.get(timeout=5)
            queue.close()
            raise RuntimeError(f"Isolated {name} fit failed: {details}")
        result = queue.get(timeout=5)
        queue.close()
        if result["status"] != "complete":
            raise RuntimeError(f"Isolated fit did not complete: {result}")
        return Emulator.load(temporary / "checkpoint")


def run_id(config: dict, stage: str, split: str, target: str, model: str, seed: int) -> str:
    """Explicit readable identity; fingerprints separately prevent accidental reuse."""
    return f"{config['experiment']['name']}__{stage}__{split}__{target}__{model}__seed{seed}"


def run_matrix(config: dict) -> list[tuple[str, str, str, int]]:
    """Neural models use every seed; deterministic classical baselines run once."""
    return [(split, target, model, seed) for split in config["experiment"]["splits"] for target in config["data"]["targets"] for model in config["experiment"]["models"] for seed in (config["training"]["seeds"] if model in NEURAL else config["training"]["seeds"][:1])]


def claim_run(directory: Path, fingerprint: dict, retry_failed: bool) -> dict | None:
    """Claim a run exclusively or validate an already complete artifact set."""
    if directory.exists():
        status_path = directory / "status.json"
        if not status_path.is_file():
            raise FileExistsError(f"Run directory has no status; inspect it manually: {directory}")
        status = read_json(status_path)
        if status.get("fingerprint") != fingerprint:
            raise ValueError(f"Run fingerprint changed; choose a new experiment name: {directory}")
        if status["status"] == "complete":
            for item in status["artifacts"]:
                path = directory / item["path"]
                if not path.is_file() or file_hash(path) != item["sha256"]:
                    raise ValueError(f"Completed artifact missing or modified: {path}")
            for item in status.get("prediction_files", []):
                path = directory.parents[1] / item["path"]
                if not path.is_file() or file_hash(path) != item["sha256"]:
                    raise ValueError(f"Completed prediction missing or modified: {path}")
            return status
        if status["status"] == "running" or not retry_failed:
            raise FileExistsError(f"Run {status['status']}; use explicit --retry-failed after investigation: {directory}")
        # Failed-run artifacts are preserved, never erased or misidentified as complete.
        existing_attempts = list(directory.glob("attempt_*.json"))
        write_json(directory / f"attempt_{len(existing_attempts) + 1}.json", status)
    else:
        directory.mkdir(parents=True, exist_ok=False)
    lock = directory / ".active"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(time.time()))
    write_json(directory / "status.json", {"status": "running", "fingerprint": fingerprint})
    return None


def validation_score(model, validation, config: dict, scenario: str) -> dict:
    """Use only inner validation and exact benchmark averaging conventions."""
    predicted = model.predict(validation)
    truth = prediction_array(validation.target, validation, scenario)
    pred = prediction_array(predicted, validation, scenario)
    return benchmark_metrics(period(truth, config["evaluation"]["benchmark_years"]), period(pred, config["evaluation"]["benchmark_years"]), validation.mask, tolerance=config["evaluation"]["denominator_tolerance"], global_weight=config["evaluation"]["global_weight"])


def run_one(config: dict, fingerprint: dict, stage: str, split_name: str, target: str, model_name: str, seed: int, retry_failed: bool = False) -> dict:
    """Execute a selected stage; no outer targets are loaded during selection."""
    if stage not in {"selection", "final"}:
        raise ValueError("Stage must be selection or final")
    if stage == "final":
        verify_freeze(config, fingerprint)
    output = resolve_path(config, config["experiment"]["output_root"])
    identifier = run_id(config, stage, split_name, target, model_name, seed)
    directory = output / "runs" / identifier
    previous = claim_run(directory, fingerprint, retry_failed)
    if previous is not None:
        LOGGER.info("Resuming verified completed run %s", identifier)
        return previous
    started = time.monotonic()
    fitting_seconds = 0.0
    split = config["splits"][split_name]
    root = resolve_path(config, config["data"]["root"])
    try:
        check_project_budget(config, model_name)
        train_scenarios = [s for s in split["train"] if stage != "selection" or s != split["validation"]]
        raw = collect_samples(root, train_scenarios, target, config)
        preprocessor = Preprocessor(config).fit(raw)
        train = preprocessor.transform(raw)
        joblib.dump(preprocessor, directory / "preprocessor.joblib")
        write_json(directory / "preprocessing.json", preprocessor.summary())
        write_json(directory / "configuration.json", protocol_configuration(config))
        write_json(directory / "environment.json", environment_versions())
        validation = None
        if stage == "selection":
            validation_raw = collect_samples(root, [split["validation"]], target, config)
            validation = preprocessor.transform(validation_raw)
            best_score, selected, candidate_records = np.inf, None, []
            model_budget_start = time.monotonic()
            for index, parameters in enumerate(model_candidates(model_name, config)):
                candidate_dir = directory / f"candidate_{index}"
                model = make_model(model_name, config, seed, parameters, None, candidate_dir)
                remaining = config["models"][model_name]["budget_seconds"] - (time.monotonic() - model_budget_start) if model_name in {"rf", "gp"} else None
                fitting_started = time.monotonic()
                try:
                    model = fit_with_deadline(model, train, validation, model_name, config, candidate_dir, remaining)
                finally:
                    fitting_seconds += time.monotonic() - fitting_started
                scores = validation_score(model, validation, config, split["validation"])
                score = scores["nrmse_spatial"]
                # The fixed fallback avoids metric selection on a near-zero denominator.
                rank = score if score is not None else scores["spatial_rmse"]
                candidate_records.append({"parameters": parameters, "scores": scores, "ranking_metric": "nrmse_spatial" if score is not None else "spatial_rmse_undefined_normalization"})
                if rank < best_score:
                    best_score = rank
                    selected = {"parameters": parameters, "epochs": getattr(model, "selected_epochs", None), "validation_scores": scores}
                    model.save(directory / "checkpoint")
                del model
                if model_name in NEURAL:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
            if selected is None:
                raise RuntimeError("No valid inner-validation candidate")
            record = {"model": model_name, "seed": seed, "split": split_name, "target": target, **selected, "candidates": candidate_records}
            write_json(directory / "selection.json", record)
            prediction_files = []
        elif stage == "final":
            selection_path = output / "runs" / run_id(config, "selection", split_name, target, model_name, seed) / "selection.json"
            record = read_json(selection_path)
            model = make_model(model_name, config, seed, record["parameters"], record["epochs"], directory / "training")
            remaining = None
            if model_name in {"rf", "gp"}:
                selection_status = read_json(selection_path.parent / "status.json")
                remaining = config["models"][model_name]["budget_seconds"] - selection_status["fitting_seconds"]
            fitting_started = time.monotonic()
            try:
                model = fit_with_deadline(model, train, None, model_name, config, directory / "training", remaining)
            finally:
                fitting_seconds += time.monotonic() - fitting_started
            model.save(directory / "checkpoint")
            prediction_files = []
            for scenario in split["test"]:
                test_raw = collect_samples(root, [scenario], target, config)
                test = preprocessor.transform(test_raw)
                variants = [(model_name, None)]
                if model_name == "residual_bounded":
                    variants += [("residual_bounded_gated", config["gate"]["primary_strength"])]
                    variants += [(f"residual_bounded_gate_{strength:g}", strength) for strength in config["gate"]["sensitivity_strengths"]]
                for variant, strength in variants:
                    values = model.predict(test, gated=True, strength=strength) if strength is not None else model.predict(test)
                    relative = Path("predictions") / identifier / f"{scenario}__{variant}.nc"
                    metadata = {"run_id": identifier, "split": split_name, "scenario": scenario, "target": target, "model": variant, "seed": seed, "protocol": config["training"]["protocol"], "units": "mm/day" if target in {"pr", "pr90"} else "K", "reference_members": test_raw.members[scenario], "configuration_hash": fingerprint["configuration_hash"], "source_hash": fingerprint["source_hash"], "dataset_hash": fingerprint["dataset_hash"], "manifest_hash": fingerprint["manifest_hashes"][f"{split_name}_final.csv"], "checkpoint": str(directory / "checkpoint"), "target_scale": train.scale, "gate_strength": strength, "support_label": "measured_input_extrapolation" if np.any(test.distance > 0) else "scenario_holdout_no_marginal_violation"}
                    save_predictions(output / relative, values, test, scenario, metadata)
                    prediction_files.append({"path": relative.as_posix(), "sha256": file_hash(output / relative)})
            del model
            if model_name in NEURAL:
                import tensorflow as tf
                tf.keras.backend.clear_session()
        else:
            raise ValueError("Stage must be selection or final")
        artifacts = [{"path": p.relative_to(directory).as_posix(), "sha256": file_hash(p)} for p in directory.rglob("*") if p.is_file() and p.name not in {"status.json", ".active"}]
        status = {"status": "complete", "stage": stage, "run_id": identifier, "fingerprint": fingerprint, "selection": record, "prediction_files": prediction_files, "artifacts": artifacts, "runtime_seconds": time.monotonic() - started, "fitting_seconds": fitting_seconds}
        write_json(directory / "status.json", status)
        return status
    except Exception as exc:
        # Deliberate orchestration boundary: persist details and re-raise, never hide failure.
        write_json(directory / "status.json", {"status": "incomplete_budget" if isinstance(exc, FitBudgetExceeded) else "failed", "stage": stage, "run_id": identifier, "fingerprint": fingerprint, "exception": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "runtime_seconds": time.monotonic() - started, "fitting_seconds": fitting_seconds})
        raise
    finally:
        (directory / ".active").unlink(missing_ok=True)


def freeze_protocol(config: dict, fingerprint: dict) -> dict:
    """Require every registered selection run before locking test-facing execution."""
    if config["experiment"]["mode"] != "core":
        raise ValueError("Smoke experiments cannot freeze a final research protocol")
    output = resolve_path(config, config["experiment"]["output_root"])
    selections, hashes = [], {}
    for split, target, model, seed in run_matrix(config):
        directory = output / "runs" / run_id(config, "selection", split, target, model, seed)
        status = read_json(directory / "status.json")
        if status["status"] != "complete" or status["fingerprint"] != fingerprint:
            raise ValueError(f"Selection incomplete or changed: {directory}")
        claim_run(directory, fingerprint, False)
        path = directory / "selection.json"
        hashes[str(path.relative_to(output))] = file_hash(path)
        selections.append(read_json(path))
    eligible = config["statistics"]["comparator_models"]
    incumbents = {f"{split}/{target}": select_incumbent([r for r in selections if r["split"] == split and r["target"] == target], eligible) for split in config["experiment"]["splits"] for target in config["data"]["targets"]}
    lock = {"fingerprint": fingerprint, "selection_hashes": hashes, "incumbents": incumbents}
    path = resolve_path(config, config["experiment"]["freeze_file"])
    if path.exists() and read_json(path) != lock:
        raise ValueError("Existing protocol lock differs; use a new experiment name")
    write_json(path, lock)
    return lock


def verify_freeze(config: dict, fingerprint: dict) -> dict:
    """Reject changed configurations or selections after protocol freezing."""
    lock = read_json(resolve_path(config, config["experiment"]["freeze_file"]))
    if lock["fingerprint"] != fingerprint:
        raise ValueError("Protocol configuration/source/data changed after freeze")
    output = resolve_path(config, config["experiment"]["output_root"])
    for path, checksum in lock["selection_hashes"].items():
        if file_hash(output / path) != checksum:
            raise ValueError(f"Selection artifact changed after freeze: {path}")
    return lock


def run_smoke(config: dict, fingerprint: dict, retry_failed: bool = False) -> None:
    """Training-only reduced-data fits; never load an outer-test target."""
    output = resolve_path(config, config["experiment"]["output_root"])
    root = resolve_path(config, config["data"]["root"])
    for split, target, name, seed in run_matrix(config):
        identifier = run_id(config, "smoke", split, target, name, seed)
        directory = output / "runs" / identifier
        if claim_run(directory, fingerprint, retry_failed) is not None:
            continue
        try:
            smoke = config["smoke"]
            raw = collect_samples(root, config["splits"][split]["train"], target, config, smoke["endpoints"], (smoke["latitude_points"], smoke["longitude_points"]))
            pipeline = Preprocessor(config).fit(raw)
            data = pipeline.transform(raw)
            model = make_model(name, config, seed, model_candidates(name, config)[0], config["training"]["max_epochs"], directory)
            model = fit_with_deadline(model, data, None, name, config, directory)
            predicted = model.predict(data)
            if predicted.shape != data.target.shape:
                raise ValueError("Smoke output shape mismatch")
            model.save(directory / "checkpoint")
            write_json(directory / "smoke.json", {"shape": predicted.shape, "training_only": True, "preprocessing": pipeline.summary()})
            artifacts = [{"path": p.relative_to(directory).as_posix(), "sha256": file_hash(p)} for p in directory.rglob("*") if p.is_file() and p.name not in {"status.json", ".active"}]
            write_json(directory / "status.json", {"status": "complete", "stage": "smoke", "fingerprint": fingerprint, "artifacts": artifacts, "prediction_files": []})
        except Exception as exc:
            write_json(directory / "status.json", {"status": "failed", "fingerprint": fingerprint, "message": str(exc), "traceback": traceback.format_exc()})
            raise
        finally:
            (directory / ".active").unlink(missing_ok=True)


def run_experiments(config: dict, stage: str = "all", retry_failed: bool = False, model_filter: set[str] | None = None) -> None:
    """Execute only on explicit invocation; finalize selection before any outer test."""
    fingerprint = preflight(config)
    if config["experiment"]["mode"] == "smoke":
        run_smoke(config, fingerprint, retry_failed)
        return
    matrix = run_matrix(config)
    selected_matrix = [row for row in matrix if model_filter is None or row[2] in model_filter]
    if stage in {"all", "selection"}:
        for split, target, name, seed in selected_matrix:
            run_one(config, fingerprint, "selection", split, target, name, seed, retry_failed)
    if stage in {"all", "freeze"}:
        freeze_protocol(config, fingerprint)
    if stage in {"all", "final"}:
        verify_freeze(config, fingerprint)
        for split, target, name, seed in selected_matrix:
            run_one(config, fingerprint, "final", split, target, name, seed, retry_failed)


def check_project_budget(config: dict, model_name: str) -> None:
    """Refuse new neural fits once recorded project fitting time reaches its cap.

    This counts all recorded attempts in the output root. A fit already in progress
    may finish within its own deadline; concurrent launch accounting is not global.
    """
    if model_name not in NEURAL:
        return
    output = resolve_path(config, config["experiment"]["output_root"])
    elapsed = 0.0
    for path in output.glob("runs/*/*.json"):
        if path.name != "status.json" and not path.name.startswith("attempt_"):
            continue
        status = read_json(path)
        if status.get("selection", {}).get("model") in NEURAL or any(f"__{name}__" in path.parent.name for name in NEURAL):
            elapsed += status.get("fitting_seconds", 0.0)
    if elapsed >= config["training"]["project_budget_seconds"]:
        raise FitBudgetExceeded("Recorded neural project fitting budget exhausted; inspect statuses before authorizing a new budget")
