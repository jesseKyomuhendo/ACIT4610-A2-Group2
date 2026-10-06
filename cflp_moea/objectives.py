"""Objective evaluation (computed only after a feasible assignment is produced).

f1 = sum_i F_i y_i          (facility-opening cost, OR-Library "fixed cost" field)
f2 = sum_i sum_j C_ij x_ij  (customer-allocation cost; C_ij NOT multiplied by d_j)

``x`` comes from ``representation.decode`` and holds demand shares, so a customer split
over several facilities (cap41/42 only) is charged C_ij * (q_ij / d_j) by each of them.

``Evaluator`` wraps decode + objectives and counts every evaluation, so both MOEAs are
stopped by the same budget of objective evaluations.
"""
from __future__ import annotations

import numpy as np

from cflp_moea.data_loader import CFLPInstance
from cflp_moea.representation import Solution, decode


def facility_cost(instance: CFLPInstance, sol: Solution) -> float:
    """Objective 1: total fixed cost of the opened facilities."""
    return float(instance.fixed_cost[sol.y].sum())


def allocation_cost(instance: CFLPInstance, sol: Solution) -> float:
    """Objective 2: total allocation cost of the customer assignment."""
    return float((instance.alloc_cost * sol.x).sum())


def objectives(instance: CFLPInstance, sol: Solution) -> np.ndarray:
    """Return the objective vector [f1, f2] (both minimised)."""
    return np.array([facility_cost(instance, sol), allocation_cost(instance, sol)])


class Evaluator:
    """Decode, repair and evaluate chromosomes while counting evaluations."""

    def __init__(self, instance: CFLPInstance):
        self.instance = instance
        self.n_evals = 0

    def __call__(self, chromosome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (repaired chromosome, objective vector [f1, f2])."""
        sol = decode(self.instance, chromosome)
        self.n_evals += 1
        return sol.y, objectives(self.instance, sol)
