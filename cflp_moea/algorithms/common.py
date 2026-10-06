"""Shared multi-objective building blocks written by hand (no library calls):
Pareto dominance, non-dominated filtering, binary tournament selection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """a dominates b: no worse in every objective and better in at least one."""
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_mask(F: np.ndarray) -> np.ndarray:
    """True for rows of F (one row per solution) that no other row dominates."""
    n = len(F)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(F[j], F[i]):
                mask[i] = False
                break
    return mask


def unique_non_dominated(X: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Non-dominated solutions with duplicate objective vectors removed."""
    mask = non_dominated_mask(F)
    X, F = X[mask], F[mask]
    _, keep = np.unique(F, axis=0, return_index=True)
    keep.sort()
    return X[keep], F[keep]


def binary_tournament(score: np.ndarray, n_select: int, rng: np.random.Generator) -> np.ndarray:
    """Pick two at random, keep the one with the lower score; returns chosen indices."""
    a = rng.integers(len(score), size=n_select)
    b = rng.integers(len(score), size=n_select)
    return np.where(score[a] <= score[b], a, b)


@dataclass
class RunResult:
    """What every MOEA run returns."""
    X: np.ndarray          # final non-dominated chromosomes
    F: np.ndarray          # their objectives [f1, f2]
    n_evals: int           # objective evaluations used
