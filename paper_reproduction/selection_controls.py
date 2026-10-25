"""Simple competitors with explicit fold scores and no outer-target selection."""
import numpy as np
from climate_emulator.models.pattern_scaling import PatternScaling
from climate_emulator.models.ridge import RidgeEmulator


class LinearBlend:
    def __init__(self, alpha, ridge_weight):
        if not 0 <= ridge_weight <= 1:
            raise ValueError('Scalar ridge weight must lie in [0,1]')
        self.alpha, self.ridge_weight = alpha, ridge_weight

    def fit(self, data):
        self.lps = PatternScaling().fit(data)
        self.ridge = RidgeEmulator(self.alpha).fit(data)
        return self

    def predict(self, data):
        return (1-self.ridge_weight)*self.lps.predict(data)+self.ridge_weight*self.ridge.predict(data)


def grid(settings, alphas):
    values = settings['scalar_weights']
    if (not values or len(set(values)) != len(values) or
            any(not np.isfinite(v) or not 0 <= v <= 1 for v in values) or not {0., 1.} <= set(values)):
        raise ValueError('scalar_weights must be distinct finite values in [0,1], including both endpoints')
    return [dict(family='linear_blend', alpha=a, ridge_weight=w) for a in alphas for w in values]


def choose_single(summaries, first=0, worst_weight=.25):
    count = len(summaries[0][0])
    scores, details = [], []
    for i in range(first, count):
        ratios = np.array([np.linalg.norm(s[0][i])/max(np.linalg.norm(s[0][0]), 1e-12) for s in summaries])
        score = float(ratios.mean()+worst_weight*ratios.max())
        scores.append((score, i))
        details.append(dict(candidate=i, spatial_ratios=ratios.tolist(), objective=score))
    index = min(scores)[1]
    return dict(weights=np.eye(count)[index].tolist(), selected_index=index,
                candidate_objectives=scores, ranking=sorted(details, key=lambda x: (x['objective'], x['candidate'])),
                interpretation='Training-side selection; fold scores are reused tuning evidence')
