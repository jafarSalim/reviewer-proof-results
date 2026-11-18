"""Training the same network with different random seeds gives different results.

Report the mean and spread over several seeds (or a confidence interval of the mean), never
the best run. `best_of` shows how much "pick the best of k seeds" inflates a result.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def summary(x, conf: float = 0.95):
    x = np.asarray(x, float)
    m, s, n = x.mean(), x.std(ddof=1), len(x)
    h = stats.t.ppf(1 - (1 - conf) / 2, n - 1) * s / np.sqrt(n)
    return dict(mean=m, std=s, n=n, ci_low=m - h, ci_high=m + h, min=x.min(), max=x.max())


def best_of(x, k: int, lower_is_better: bool = True, draws: int = 20000, seed: int = 0):
    """Distribution of the best of k runs drawn from the observed runs (with replacement)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    s = rng.choice(x, (draws, k))
    return s.min(1) if lower_is_better else s.max(1)
