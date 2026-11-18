"""Error rates measured by Monte Carlo: how sure are you?

A simulated BER is an estimate k/n from n bits with k errors. Its uncertainty depends on
the number of *errors*, not the number of bits: with k errors the relative 95 %
half-width is roughly 1.96/sqrt(k). Report an interval, and simulate until you have
enough errors (100 errors gives about +-20 %).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def clopper_pearson(k: int, n: int, conf: float = 0.95):
    """Exact (conservative) binomial interval."""
    a = 1 - conf
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def wilson(k: int, n: int, conf: float = 0.95):
    """Wilson score interval: close to nominal coverage, never outside [0, 1]."""
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def wald(k: int, n: int, conf: float = 0.95):
    """The textbook p +- z sqrt(p(1-p)/n): breaks down for few errors (shown in fig. 1)."""
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    h = z * math.sqrt(p * (1 - p) / n)
    return p - h, p + h


def rule_of_three(n: int) -> float:
    """Zero errors in n bits: the 95 % upper bound on the BER is about 3/n."""
    return 3.0 / n


def bits_needed(ber: float, rel_halfwidth: float = 0.2, conf: float = 0.95) -> int:
    """Bits to simulate so that the 95 % interval is +-rel_halfwidth of the BER."""
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    errors = (z / rel_halfwidth) ** 2 * (1 - ber)
    return int(math.ceil(errors / ber))


def coverage(method, p: float, n: int, trials: int = 20000, seed: int = 0) -> float:
    """Fraction of simulated experiments whose interval contains the true p."""
    rng = np.random.default_rng(seed)
    ks = rng.binomial(n, p, trials)
    hit = 0
    for k in ks:
        lo, hi = method(int(k), n)
        hit += lo <= p <= hi
    return hit / trials
