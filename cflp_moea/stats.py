"""Statistical comparison of the MOEAs over >= 10 independent runs
(mean/std/best/worst HV, Mann-Whitney U test via scipy.stats, A12 effect size).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def summary(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": v.mean(), "std": v.std(ddof=1), "best": v.max(), "worst": v.min()}


def a12(a, b) -> float:
    """Vargha-Delaney A12: chance that a value from a is larger than one from b (0.5 = equal)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    greater = (a[:, None] > b[None, :]).sum()
    ties = (a[:, None] == b[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(a) * len(b)))


def compare(a, b, alpha: float = 0.05) -> dict:
    """Two-sided Mann-Whitney U test on HV values of two algorithms (larger HV is better)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.all(a == a[0]) and np.all(b == a[0]):
        p = 1.0                                   # identical results, nothing to test
    else:
        p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    effect = a12(a, b)
    if p >= alpha:
        winner = "no significant difference"
    else:
        winner = "first" if effect > 0.5 else "second"
    return {"p_value": p, "a12": effect, "winner": winner}
