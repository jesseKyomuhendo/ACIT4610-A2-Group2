"""f1 = sum F_i y_i (opening cost), f2 = sum C_ij x_ij (allocation cost, not times demand)."""
from __future__ import annotations

import numpy as np

from cflp_moea.data_loader import CFLPInstance
from cflp_moea.representation import Solution, decode


def facility_cost(instance: CFLPInstance, sol: Solution) -> float:
    return float(instance.fixed_cost[sol.y].sum())


def allocation_cost(instance: CFLPInstance, sol: Solution) -> float:
    # split customers pay each facility its share
    return float((instance.alloc_cost * sol.x).sum())


def objectives(instance: CFLPInstance, sol: Solution) -> np.ndarray:
    return np.array([facility_cost(instance, sol), allocation_cost(instance, sol)])


class Evaluator:
    """Decode + evaluate, and count evaluations for the budget."""

    def __init__(self, instance: CFLPInstance):
        self.instance = instance
        self.n_evals = 0

    def __call__(self, chromosome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (repaired chromosome, [f1, f2])."""
        sol = decode(self.instance, chromosome)
        self.n_evals += 1
        return sol.y, objectives(self.instance, sol)
