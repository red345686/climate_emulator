"""Explicit spatial/global sufficient statistics; no outer-label selection."""
import numpy as np

ORIGINAL_ALPHAS = [0.1, 1.0, 10.0, 100.0]
ORIGINAL_WEIGHTS = [0., .25, .5, .75, 1.]
ALPHAS = [.03, .1, .3, 1., 3., 10., 30., 100., 300.]
WEIGHTS = sorted(set([i / 10 for i in range(11)] + ORIGINAL_WEIGHTS))


def summary(prediction, truth, weights):
    error = np.asarray(prediction) - np.asarray(truth)
    if error.ndim != 2 or not np.isfinite(error).all():
        raise ValueError('Expected finite years x valid-cells arrays')
    return np.r_[error.mean(axis=0) * np.sqrt(weights), error @ weights]


def scores(value, cells, denominator):
    spatial = float(np.linalg.norm(value[:cells]))
    global_error = float(np.sqrt(np.mean(value[cells:] ** 2)))
    return dict(spatial_rmse=spatial, global_rmse=global_error,
                nrmse_total=(spatial + 5 * global_error) / denominator if denominator > 1e-10 else None)


def objective(lps, ridge, weight, cells):
    ratios = [np.linalg.norm(((1-weight)*l+weight*r)[:cells]) /
              max(np.linalg.norm(l[:cells]), 1e-12) for l, r in zip(lps, ridge)]
    return float(np.mean(ratios) + .25 * max(ratios))


def select(lps, branches, cells):
    """Original order: explicit LPS, then alpha-major original 4 x 5 grid."""
    candidates = [(objective(lps, lps, 0., cells), 0, None, 0.)]
    for alpha in ORIGINAL_ALPHAS:
        for weight in ORIGINAL_WEIGHTS:
            candidates.append((objective(lps, branches[alpha], weight, cells),
                               len(candidates), alpha, weight))
    value, _, alpha, weight = min(candidates)
    return dict(alpha=alpha, weight=weight, objective=value)


def complementarity(left, right, weight, cells):
    """Weighted error inner product, NOT a Pearson correlation or significance test."""
    l, r = left[:cells], right[:cells]
    ll, rr, cross = float(l@l), float(r@r), float(l@r)
    blend = (1-weight)*l + weight*r
    actual = float(blend@blend)
    algebra = (1-weight)**2*ll + weight**2*rr + 2*weight*(1-weight)*cross
    denominator = ll + rr - 2*cross
    oracle = float(np.clip((ll-cross)/denominator, 0, 1)) if denominator > 1e-20 else None
    return dict(lps_spatial_mse=ll, ridge_spatial_mse=rr, spatial_error_inner_product=cross,
                spatial_error_cosine=cross/np.sqrt(ll*rr) if ll*rr > 1e-30 else None,
                cross_term=2*weight*(1-weight)*cross, blend_spatial_mse=actual,
                identity_absolute_error=abs(actual-algebra),
                spatial_mse_gain_vs_weighted_branch_mse=(1-weight)*ll+weight*rr-actual,
                outer_oracle_weight_diagnostic_only=oracle)
