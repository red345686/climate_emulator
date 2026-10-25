import numpy as np

def folds(split):
    pool = list(split["train"])
    if set(pool) & set(split["test"]):
        raise ValueError("Outer test scenario appears in training pool")
    held = [s for s in pool if s.startswith("ssp")]
    if len(held) < 2:
        raise ValueError("At least two future training scenarios are required")
    return [(s, [other for other in pool if other != s]) for s in held]


def error_summary(values, data, years):
    """Sufficient statistics for exact period spatial/global RMSE of any blend."""
    positions = [i for i, (_, y) in enumerate(data.keys) if years[0] <= y <= years[1]]
    selected_keys = [data.keys[i] for i in positions]
    if (len({s for s, _ in selected_keys}) != 1 or
            [y for _, y in selected_keys] != list(range(years[0], years[1] + 1))):
        raise ValueError("Incomplete validation benchmark period")
    error = values[positions][:, data.mask] - data.target[positions][:, data.mask]
    if not np.isfinite(error).all():
        raise ValueError("Nonfinite validation errors on training mask")
    weights = data.weights[data.mask]
    weights = weights / weights.sum()
    return error.mean(axis=0) * np.sqrt(weights), error @ weights

