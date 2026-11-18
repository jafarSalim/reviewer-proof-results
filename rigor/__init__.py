"""rigor: small, tested helpers for results that survive peer review.

ber        confidence intervals for error rates, bits needed per point, rule of three
seeds      seed-to-seed spread, honest reporting of repeated training runs
leakage    group-aware splitting (no channel / patient / session in both train and test)
compare    paired significance tests: McNemar, 5x2cv paired t (Dietterich), corrected
           resampled t (Nadeau and Bengio), Holm-Bonferroni for many comparisons
"""
__version__ = "1.0.0"
