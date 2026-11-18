import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rigor import ber, compare, leakage, seeds  # noqa: E402


def test_intervals_contain_estimate_and_rule_of_three():
    for f in (ber.wilson, ber.clopper_pearson):
        lo, hi = f(37, 100000)
        assert lo < 37 / 100000 < hi
    lo, hi = ber.clopper_pearson(0, 1_000_000)
    assert lo == 0 and abs(hi - 3.69e-6) < 2e-7       # exact bound is ln(40)/n ≈ 3.69/n
    assert abs(ber.rule_of_three(1_000_000) - 3e-6) < 1e-12


def test_bits_needed_matches_100_errors_rule():
    n = ber.bits_needed(1e-4, 0.2)
    assert 95 / 1e-4 < n < 97 / 1e-4                   # (1.96/0.2)^2 ≈ 96 errors


def test_group_split_has_no_overlap():
    g = np.repeat(np.arange(50), 10)
    tr, te = leakage.group_split(g, 0.2, seed=1)
    assert leakage.overlap(g, tr, te) == 0.0
    tr2, te2 = leakage.random_split(len(g), 0.2, seed=1)
    assert leakage.overlap(g, tr2, te2) > 0.9


def test_mcnemar_and_holm():
    a = np.array([1] * 90 + [0] * 10, bool)
    assert compare.mcnemar(a, a) == 1.0
    b = a.copy(); b[90:] = True                        # B fixes all 10 errors of A
    assert compare.mcnemar(a, b) < 0.01
    assert compare.holm([0.001, 0.02, 0.04, 0.5]).tolist() == [True, False, False, False]


def test_seed_summary_and_best_of():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    s = seeds.summary(x)
    assert s["mean"] == 2.5 and s["ci_low"] < 2.5 < s["ci_high"]
    assert np.median(seeds.best_of(x, 3)) <= np.median(seeds.best_of(x, 1))
