"""Leakage: when the test set is not new.

Deep-learning channel estimators are often trained on many noisy observations of the same
channel realization. A random split puts observations of one channel in both training and
test sets; the network then recognises the channel instead of estimating it. Split by
*group* (channel realization, user, session, patient) instead.
"""
from __future__ import annotations

import numpy as np


def random_split(n: int, test_frac: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(round(test_frac * n))
    return idx[k:], idx[:k]


def group_split(groups, test_frac: float, seed: int = 0):
    """Train/test indices such that no group appears in both."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    g = rng.permutation(np.unique(groups))
    k = int(round(test_frac * len(g)))
    test_g = set(g[:k].tolist())
    is_test = np.array([x in test_g for x in groups])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def overlap(groups, train_idx, test_idx) -> float:
    """Fraction of test samples whose group also appears in the training set."""
    groups = np.asarray(groups)
    tr = set(groups[train_idx].tolist())
    return float(np.mean([groups[i] in tr for i in test_idx]))
