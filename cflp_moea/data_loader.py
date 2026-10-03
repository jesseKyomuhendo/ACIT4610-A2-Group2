"""Data loading: parse OR-Library capacitated warehouse location files (cap*.txt).

Only reads values from the file; nothing is generated or modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "orlib"


@dataclass(frozen=True)
class CFLPInstance:
    name: str
    m: int                      # number of candidate facilities
    n: int                      # number of customers
    capacity: np.ndarray        # S_i, shape (m,)
    fixed_cost: np.ndarray      # F_i, shape (m,)   -> Objective 1
    demand: np.ndarray          # d_j, shape (n,)
    alloc_cost: np.ndarray      # C_ij, shape (m, n) -> Objective 2 (already includes demand)

    @property
    def total_demand(self) -> float:
        return float(self.demand.sum())


def load_instance(name_or_path: str | Path) -> CFLPInstance:
    """Load e.g. 'cap41' (looked up in data/orlib) or a direct path to a cap*.txt file."""
    path = Path(name_or_path)
    if not path.suffix:
        path = DATA_DIR / f"{path.name}.txt"
    tokens = path.read_text().split()
    pos = 0

    def nxt() -> float:
        nonlocal pos
        val = float(tokens[pos])
        pos += 1
        return val

    m, n = int(nxt()), int(nxt())
    capacity = np.empty(m)
    fixed_cost = np.empty(m)
    for i in range(m):
        capacity[i] = nxt()
        fixed_cost[i] = nxt()

    demand = np.empty(n)
    alloc_cost = np.empty((m, n))
    for j in range(n):
        demand[j] = nxt()
        for i in range(m):
            alloc_cost[i, j] = nxt()

    if pos != len(tokens):
        raise ValueError(f"{path.name}: {len(tokens) - pos} unread tokens - unexpected format")
    if capacity.sum() < demand.sum():
        raise ValueError(f"{path.name}: total capacity < total demand (infeasible instance)")

    return CFLPInstance(path.stem, m, n, capacity, fixed_cost, demand, alloc_cost)


def load_optimal_values(path: Path = DATA_DIR / "capopt.txt") -> dict[str, float]:
    """Known single-objective optima (f1+f2) from capopt.txt - sanity reference only."""
    opt: dict[str, float] = {}
    for line in path.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[0].startswith("cap"):
            opt[parts[0]] = float(parts[1])
    return opt
