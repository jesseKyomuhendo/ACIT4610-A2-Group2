"""Objective evaluation and Pareto dominance (both objectives are minimized).

    f1 = sum_i F_i * y_i            facility-opening cost
    f2 = sum_i sum_j C_ij * x_ij    customer-allocation cost

With the representation in src/representation.py, customer j is served by
facility chromosome[j], so f2 is simply the sum of C[chromosome[j], j].
C_ij already covers all of customer j's demand, so it is NOT multiplied by d_j.

Objectives are only calculated for feasible solutions: evaluate() refuses an
infeasible chromosome, so it must be repaired first.
"""

import numpy as np

from src.repair import is_feasible
from src.representation import open_facilities


# -----------------------------------------------------------------------------
# Objectives
# -----------------------------------------------------------------------------
def evaluate(chromosome, inst):
    """Return the objective vector [f1, f2] of a feasible chromosome."""
    if not is_feasible(chromosome, inst):
        raise ValueError("evaluate() was called on an infeasible chromosome; repair it first.")

    y = open_facilities(chromosome, inst.m)
    f1 = np.sum(inst.fixed_cost * y)
    f2 = np.sum(inst.cost[chromosome, np.arange(inst.n)])
    return np.array([f1, f2])


def evaluate_population(population, inst):
    """Return an array of shape (len(population), 2) with [f1, f2] per individual."""
    return np.array([evaluate(chromosome, inst) for chromosome in population])


# -----------------------------------------------------------------------------
# Pareto dominance
# -----------------------------------------------------------------------------
def dominates(a, b):
    """True if objective vector a dominates b (minimization).

    a dominates b when a is no worse than b in every objective
    and strictly better in at least one (Lecture 3).
    """
    return bool(np.all(a <= b) and np.any(a < b))


def dominance_matrix(points):
    """Return D with D[p, q] = True if point p dominates point q.

    Same rule as dominates(), but for all pairs at once with numpy, which is
    much faster than calling dominates() N*N times inside the MOEAs.
    """
    points = np.asarray(points)
    a = points[:, None, :]   # shape (N, 1, 2): point p
    b = points[None, :, :]   # shape (1, N, 2): point q
    no_worse_everywhere = np.all(a <= b, axis=2)
    better_somewhere = np.any(a < b, axis=2)
    return no_worse_everywhere & better_somewhere


def non_dominated_indices(points):
    """Return the indices of the points that no other point dominates.

    points: array of shape (N, 2). Duplicate points are all kept.
    """
    dominated = dominance_matrix(points).any(axis=0)   # column q: is q dominated by anyone?
    return np.flatnonzero(~dominated)


def non_dominated(points):
    """Return the non-dominated subset of points, with duplicates removed, sorted by f1."""
    points = np.asarray(points)
    front = np.unique(points[non_dominated_indices(points)], axis=0)  # unique() also sorts by f1
    return front