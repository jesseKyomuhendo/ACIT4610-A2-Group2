"""Wilcoxon signed-rank test, implemented by hand (no scipy).

Used to compare the hypervolumes of NSGA-II and SPEA2 on the same instance and
configuration. The test is PAIRED: run k of NSGA-II and run k of SPEA2 use the
same seed, so they form a pair.

Procedure
---------
1. Differences d_k = a_k - b_k for every pair; pairs with d_k = 0 are dropped.
2. Rank |d_k| from smallest (rank 1) to largest; tied values get the average
   of the ranks they cover.
3. W+ = sum of ranks of positive differences (a better when larger values are better),
   W- = sum of ranks of negative differences.
4. Exact two-sided p-value: if there were no real difference, every rank would
   be equally likely to carry a + or - sign, so all 2^n sign patterns are
   equally likely. We count how many patterns give a W+ at least as extreme as
   the observed one:
       p = 2 * min( P(W+ <= observed), P(W+ >= observed) ),  capped at 1.
   The counting uses a running table of "how many patterns give each W+"
   instead of listing all 2^n patterns, so it stays fast for many runs.
With 10 pairs the smallest possible p-value is 2/1024 = 0.002.
"""

import numpy as np

from src.util.fetch_config import ALPHA


def average_ranks(values):
    """Ranks 1..n of values (smallest = 1); ties get the average of their ranks."""
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1   # average of ranks start+1..end+1
        start = end + 1
    return ranks


def wilcoxon_signed_rank(a, b):
    """Paired two-sided Wilcoxon signed-rank test with an exact p-value.

    Returns (w_plus, w_minus, p_value, n_used), where n_used is the number of
    pairs with a non-zero difference.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[~np.isclose(diff, 0.0, rtol=0.0, atol=1e-12)]   # drop zero differences
    n = len(diff)
    if n == 0:
        return 0.0, 0.0, 1.0, 0

    ranks = average_ranks(np.abs(diff))
    w_plus = float(ranks[diff > 0].sum())
    w_minus = float(ranks[diff < 0].sum())

    # Count sign patterns per possible W+. Ranks can be x.5 because of ties,
    # so all ranks are doubled to make them whole numbers.
    doubled = np.round(ranks * 2).astype(int)
    counts = np.zeros(doubled.sum() + 1)
    counts[0] = 1                                   # no ranks yet: one pattern, W+ = 0
    for r in doubled:
        with_plus = np.zeros_like(counts)
        with_plus[r:] = counts[:-r]                 # this rank gets a + sign: W+ grows by r
        counts = counts + with_plus                 # ... or a - sign: W+ unchanged

    probability = counts / counts.sum()             # every pattern equally likely
    observed = int(round(w_plus * 2))
    p_lower = probability[:observed + 1].sum()      # P(W+ <= observed)
    p_upper = probability[observed:].sum()          # P(W+ >= observed)
    p_value = min(1.0, 2 * min(p_lower, p_upper))
    return w_plus, w_minus, float(p_value), n


def compare(hv_a, hv_b, name_a="NSGA-II", name_b="SPEA2"):
    """Compare two lists of hypervolumes (larger is better) from paired runs.

    Returns a dict with the test result and which algorithm is significantly
    better at significance level ALPHA (or "no significant difference").
    """
    w_plus, w_minus, p_value, n_used = wilcoxon_signed_rank(hv_a, hv_b)
    if p_value < ALPHA:
        winner = name_a if w_plus > w_minus else name_b
    else:
        winner = "no significant difference"
    return {"w_plus": w_plus, "w_minus": w_minus, "p_value": p_value,
            "n_pairs": n_used, "better": winner}