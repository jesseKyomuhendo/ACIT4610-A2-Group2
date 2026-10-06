"""Wilcoxon signed-rank test implemented by hand without scipy.

It compares the hypervolumes of NSGA-II and SPEA2, paired by run because run k of both uses the same seed.
"""

import numpy as np

from src.util.fetch_config import ALPHA


def average_ranks(values):
    """Ranks values from 1 for the smallest, where tied values share the average of their ranks."""
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    start = 0
    while start < len(values):
        # Find the group of equal values that starts here
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1   # average rank of the group
        start = end + 1
    return ranks


def wilcoxon_signed_rank(a, b):
    """Paired two-sided Wilcoxon signed-rank test with an exact p-value.

    Returns W plus, W minus, the p-value and the number of pairs with a non-zero difference.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[~np.isclose(diff, 0.0, rtol=0.0, atol=1e-12)]   # drop zero differences
    n = len(diff)
    if n == 0:
        return 0.0, 0.0, 1.0, 0

    # Rank the size of each difference, then add up the ranks of positive and negative ones
    ranks = average_ranks(np.abs(diff))
    w_plus = float(ranks[diff > 0].sum())
    w_minus = float(ranks[diff < 0].sum())

    # Without a real difference every rank is equally likely to be positive or negative
    # The table counts how many of the 2^n sign patterns give each possible W plus
    doubled = np.round(ranks * 2).astype(int)       # doubling turns tied half ranks into whole numbers
    counts = np.zeros(doubled.sum() + 1)
    counts[0] = 1                                   # before any rank there is one pattern with W plus 0
    for r in doubled:
        with_plus = np.zeros_like(counts)
        with_plus[r:] = counts[:-r]                 # this rank is positive, so W plus grows by r
        counts = counts + with_plus                 # or negative, so W plus stays the same

    probability = counts / counts.sum()             # every pattern is equally likely
    observed = int(round(w_plus * 2))
    p_lower = probability[:observed + 1].sum()      # chance of W plus at most the observed value
    p_upper = probability[observed:].sum()          # chance of W plus at least the observed value
    # Two-sided p-value, so twice the smaller tail and never above 1
    # With 10 pairs the smallest possible value is 2/1024, about 0.002
    p_value = min(1.0, 2 * min(p_lower, p_upper))
    return w_plus, w_minus, float(p_value), n


def compare(hv_a, hv_b, name_a="NSGA-II", name_b="SPEA2"):
    """Compares hypervolumes of paired runs, where larger is better.

    Returns the test result and which algorithm is significantly better at level ALPHA.
    """
    w_plus, w_minus, p_value, n_used = wilcoxon_signed_rank(hv_a, hv_b)
    # A larger W plus means the first algorithm usually had the higher hypervolume
    if p_value < ALPHA:
        winner = name_a if w_plus > w_minus else name_b
    else:
        winner = "no significant difference"
    return {"w_plus": w_plus, "w_minus": w_minus, "p_value": p_value,
            "n_pairs": n_used, "better": winner}