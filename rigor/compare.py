"""Is model A really better than model B?

The naive approach, a t-test on the k cross-validation fold scores, ignores that the folds
share training data; it finds "significant" differences between identical models far more
often than 5 % (Dietterich, Neural Computation 1998; Nadeau and Bengio, Machine Learning 2003).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def naive_t(a, b):
    """Paired t-test on per-fold scores (the common mistake)."""
    return float(stats.ttest_rel(a, b).pvalue)


def corrected_resampled_t(a, b, n_train: int, n_test: int):
    """Nadeau-Bengio corrected resampled t-test for repeated train/test splits."""
    d = np.asarray(a, float) - np.asarray(b, float)
    J = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var + 1e-300)
    return float(2 * stats.t.sf(abs(t), J - 1))


def cv_5x2_t(diffs):
    """Dietterich's 5x2cv paired t-test. diffs: 5-by-2 array of score differences."""
    d = np.asarray(diffs, float)
    s2 = ((d - d.mean(1, keepdims=True)) ** 2).sum(1)
    t = d[0, 0] / np.sqrt(s2.mean() + 1e-300)
    return float(2 * stats.t.sf(abs(t), 5))


def mcnemar(correct_a, correct_b):
    """Exact McNemar test on one test set: only the disagreements matter."""
    a = np.asarray(correct_a, bool)
    b = np.asarray(correct_b, bool)
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))


def holm(pvalues, alpha: float = 0.05):
    """Holm-Bonferroni: which of many hypotheses survive at family-wise level alpha."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    reject = np.zeros(m, bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject
