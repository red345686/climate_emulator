"""Validation-selected incumbents and predeclared practical improvement rules."""
from __future__ import annotations

import numpy as np
import pandas as pd


def seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average metrics over seeds; never average prediction fields implicitly."""
    keys = ["split", "scenario", "target", "model", "reference", "metric", "protocol"]
    return metrics.groupby(keys, dropna=False).value.agg(["mean", "std", "count"]).reset_index()


def select_incumbent(records: list[dict], eligible: list[str]) -> str | None:
    """Select a baseline exclusively using mean inner-validation spatial NRMSE."""
    values: dict[str, list[float]] = {}
    for record in records:
        score = record.get("validation_scores", {}).get("nrmse_spatial")
        if record["model"] in eligible and score is not None and np.isfinite(score):
            values.setdefault(record["model"], []).append(score)
    return min(values, key=lambda name: float(np.mean(values[name]))) if values else None


def meaningful_improvement(metrics: pd.DataFrame, incumbent: str | None, target: str, threshold: float = .05, degradation: float = .05, expected_seeds: list[int] | None = None) -> dict:
    """Evaluate fixed A/B criteria; incomplete comparisons remain undetermined."""
    frame = metrics[(metrics.target == target) & (metrics.reference == "ensemble_mean")]
    def score(split: str, model: str, metric: str) -> float | None:
        rows = frame[(frame.split == split) & (frame.model == model) & (frame.metric == metric)]
        expected = {"ssp370", "ssp585"} if split == "B" else {"ssp245"}
        if set(rows.scenario) != expected or rows.value.isna().any():
            return None
        if expected_seeds is not None:
            seeds = expected_seeds if model in {"cnn_lstm", "residual_unbounded", "residual_bounded", "residual_bounded_gated"} else expected_seeds[:1]
            if any(set(group.seed) != set(seeds) or group.seed.duplicated().any() for _, group in rows.groupby("scenario")):
                return None
        return float(rows.groupby("scenario").value.mean().mean())
    if incumbent is None:
        return {"target": target, "status": "undetermined", "reason": "No finite validation-selected baseline"}
    hybrid = "residual_bounded_gated"
    pairs = [("B", "nrmse_spatial"), ("B", "nrmse_total"), ("A", "nrmse_spatial"), ("A", "nrmse_total")]
    relative = {}
    for split, metric in pairs:
        baseline, proposed = score(split, incumbent, metric), score(split, hybrid, metric)
        if baseline is None or proposed is None or baseline <= 0:
            return {"target": target, "incumbent": incumbent, "status": "undetermined", "reason": f"Missing/undefined/zero score: {split}/{metric}"}
        relative[f"{split}_{metric}"] = (baseline - proposed) / baseline
    lps, proposed = score("B", "lps", "nrmse_spatial"), score("B", hybrid, "nrmse_spatial")
    if lps is None:
        return {"target": target, "status": "undetermined", "reason": "Missing LPS comparison"}
    passed = relative["B_nrmse_spatial"] >= threshold and proposed < lps and all(relative[key] >= -degradation for key in ("B_nrmse_total", "A_nrmse_spatial", "A_nrmse_total"))
    return {"target": target, "incumbent": incumbent, "status": "criterion_met" if passed else "criterion_not_met", "relative_improvements": relative, "B_improvement_over_lps": lps - proposed, "interpretation": "Operational effect-size criterion; not statistical significance or a climate-impact threshold"}
