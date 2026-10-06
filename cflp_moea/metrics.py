"""Performance metrics: objective normalisation, hypervolume (2-D), number of
non-dominated solutions. Same normalisation + reference point for both MOEAs per instance.

Normalisation: for each instance, take the min and max of f1 and f2 over ALL final fronts
(both MOEAs, all configurations, all runs) and scale each objective to [0, 1].
Reference point: (1.1, 1.1) in the scaled space, so it is worse than every solution.
"""
from __future__ import annotations

import numpy as np

from cflp_moea.algorithms.common import unique_non_dominated

REF_POINT = np.array([1.1, 1.1])


def bounds(fronts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bound of each objective over all given fronts."""
    allF = np.vstack(fronts)
    return allF.min(axis=0), allF.max(axis=0)


def normalise(F: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    span = np.where(upper > lower, upper - lower, 1.0)
    return (F - lower) / span


def hypervolume_2d(F: np.ndarray, ref: np.ndarray = REF_POINT) -> float:
    """Area dominated by the front and bounded by ref (both objectives minimised)."""
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    _, F = unique_non_dominated(F, F)
    F = F[np.argsort(F[:, 0])]                    # f1 up, so f2 goes down
    # add one rectangle per point, between its f1 and the next point's f1
    next_f1 = np.append(F[1:, 0], ref[0])
    return float(((next_f1 - F[:, 0]) * (ref[1] - F[:, 1])).sum())


def n_non_dominated(F: np.ndarray) -> int:
    """Number of distinct non-dominated solutions in the final front."""
    return len(unique_non_dominated(F, F)[1])
