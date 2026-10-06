"""Objective evaluation and Pareto dominance, where both objectives are minimised.

Objectives are only calculated for feasible solutions, so a chromosome must be repaired first.
"""

import numpy as np

from src.repair import is_feasible
from src.representation import open_facilities


# Objectives
def evaluate(chromosome, inst):
    """Returns the objective vector [f1, f2] of a feasible chromosome."""
    if not is_feasible(chromosome, inst):
        raise ValueError("evaluate() was called on an infeasible chromosome; repair it first.")

    # f1 is the facility-opening cost, the sum of F_i over open facilities
    y = open_facilities(chromosome, inst.m)
    f1 = np.sum(inst.fixed_cost * y)
    # f2 is the customer-allocation cost, where customer j uses facility chromosome[j]
    # C_ij already covers all of the demand, so it is not multiplied by d_j
    f2 = np.sum(inst.cost[chromosome, np.arange(inst.n)])
    return np.array([f1, f2])


def evaluate_population(population, inst):
    """Returns an array with one row [f1, f2] per individual."""
    return np.array([evaluate(chromosome, inst) for chromosome in population])


# Pareto dominance
def dominates(a, b):
    """True if objective vector a dominates b, as defined in Lecture 3.

    a is no worse than b in every objective and strictly better in at least one.
    """
    return bool(np.all(a <= b) and np.any(a < b))


def dominance_matrix(points):
    """Returns D where D[p, q] is True if point p dominates point q.

    Same rule as dominates() but for all pairs at once, which is much faster inside the MOEAs.
    """
    points = np.asarray(points)
    a = points[:, None, :]   # every point p as a row
    b = points[None, :, :]   # every point q as a column
    no_worse_everywhere = np.all(a <= b, axis=2)
    better_somewhere = np.any(a < b, axis=2)
    return no_worse_everywhere & better_somewhere


def non_dominated_indices(points):
    """Returns the indices of the points that no other point dominates, keeping duplicates."""
    dominated = dominance_matrix(points).any(axis=0)   # column q is True if anyone dominates q
    return np.flatnonzero(~dominated)


def non_dominated(points):
    """Returns the non-dominated points with duplicates removed, sorted by f1."""
    points = np.asarray(points)
    front = np.unique(points[non_dominated_indices(points)], axis=0)  # unique also sorts by f1
    return front