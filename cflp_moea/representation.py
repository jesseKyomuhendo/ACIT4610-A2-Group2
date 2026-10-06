"""Chromosome representation and feasibility repair / decoding.

Shared by BOTH MOEAs (same representation + repair = fair comparison).

Representation
--------------
A chromosome is a binary vector ``y`` of length m: ``y[i] = 1`` means facility i is
opened. Customer assignments are not stored in the chromosome; they are built by the
deterministic decoder below, so every chromosome maps to exactly one solution.

Decoding / repair (``decode``)
------------------------------
1. Capacity repair: while the open capacity is below total demand, open the closed
   facility with the lowest fixed cost per unit of capacity (F_i / S_i).
2. Greedy assignment: customers are taken in order of decreasing demand (large items
   first, so they still find room) and each is assigned to the open facility with the
   lowest allocation cost C_ij that still has enough residual capacity.
3. If no open facility has room, the closed facility with room that minimises
   F_i + C_ij is opened (repair) and the customer is assigned to it.
4. Oversized customers (demand larger than every facility's capacity; only cap41/42,
   see data/orlib/README.md) are split: their demand is poured into facilities in
   order of increasing C_ij, open ones first, then closed ones by F_i + C_ij.
5. Open facilities that end up serving no customer are closed, because they would only
   add fixed cost.

The repaired ``y`` is returned too, so the MOEAs can write it back into the population
(Lamarckian repair) and the chromosome always matches the solution it represents.

The assignment is stored as a fractional matrix ``x`` of shape (m, n), where ``x[i, j]``
is the share of customer j's demand served by facility i. For single-sourced customers
each column holds one 1 and zeros elsewhere; for a split customer the column sums to 1.
With this, f2 = sum_ij C_ij x_ij charges a split customer C_ij * (q_ij / d_j).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cflp_moea.data_loader import CFLPInstance

_EPS = 1e-9


@dataclass(frozen=True)
class Solution:
    """A decoded, feasible solution."""
    y: np.ndarray       # repaired open vector, shape (m,), dtype bool
    x: np.ndarray       # assignment shares, shape (m, n), columns sum to 1


def facility_load(instance: CFLPInstance, sol: Solution) -> np.ndarray:
    """sum_j d_j x_ij for every facility i."""
    return sol.x @ instance.demand


def oversized_customers(instance: CFLPInstance) -> np.ndarray:
    """Boolean mask of customers whose demand exceeds every facility's capacity."""
    return instance.demand > instance.capacity.max()


def random_chromosome(instance: CFLPInstance, rng: np.random.Generator,
                      p_open: float = 0.5) -> np.ndarray:
    """Random open vector; each facility is opened with probability ``p_open``."""
    return rng.random(instance.m) < p_open


def decode(instance: CFLPInstance, chromosome: np.ndarray) -> Solution:
    """Repair a chromosome and build a feasible assignment (see module docstring)."""
    m, n = instance.m, instance.n
    cap, F, d, C = instance.capacity, instance.fixed_cost, instance.demand, instance.alloc_cost

    y = np.asarray(chromosome, dtype=bool).copy()

    # Step 1: make sure the open facilities can hold total demand.
    ratio_order = np.argsort(F / cap, kind="stable")
    for i in ratio_order:
        if cap[y].sum() >= d.sum():
            break
        y[i] = True

    residual = np.where(y, cap, 0.0).astype(float)
    x = np.zeros((m, n))
    big = oversized_customers(instance)

    # Step 2: customers in order of decreasing demand (stable, so ties keep index order).
    for j in np.argsort(-d, kind="stable"):
        if big[j]:
            _split_customer(j, y, residual, x, instance)
            continue
        fits = y & (residual >= d[j] - _EPS)
        if fits.any():
            i = _argmin_where(C[:, j], fits)
        else:
            # Step 3: open the cheapest closed facility that can hold this customer.
            can_open = ~y & (cap >= d[j] - _EPS)
            if not can_open.any():
                raise RuntimeError(f"{instance.name}: no facility can take customer {j}")
            i = _argmin_where(F + C[:, j], can_open)
            y[i] = True
            residual[i] = cap[i]
        x[i, j] = 1.0
        residual[i] -= d[j]

    # Step 5: close facilities that serve nobody.
    y &= x.sum(axis=1) > 0
    return Solution(y=y, x=x)


def _split_customer(j: int, y: np.ndarray, residual: np.ndarray, x: np.ndarray,
                    instance: CFLPInstance) -> None:
    """Spread an oversized customer's demand over facilities (step 4)."""
    cap, F, d, C = instance.capacity, instance.fixed_cost, instance.demand, instance.alloc_cost
    remaining = d[j]
    open_order = [i for i in np.argsort(C[:, j], kind="stable") if y[i]]
    closed_order = [i for i in np.argsort(F + C[:, j], kind="stable") if not y[i]]
    for i in open_order + closed_order:
        if remaining <= _EPS:
            break
        if not y[i]:
            y[i] = True
            residual[i] = cap[i]
        q = min(residual[i], remaining)
        if q <= _EPS:
            continue
        x[i, j] += q / d[j]
        residual[i] -= q
        remaining -= q
    if remaining > _EPS:
        raise RuntimeError(f"{instance.name}: not enough capacity for customer {j}")


def _argmin_where(values: np.ndarray, mask: np.ndarray) -> int:
    return int(np.argmin(np.where(mask, values, np.inf)))


def check_feasible(instance: CFLPInstance, sol: Solution, tol: float = 1e-6) -> list[str]:
    """Return a list of violated constraints (empty list = feasible)."""
    errors = []
    x, y = sol.x, sol.y
    if not np.allclose(x.sum(axis=0), 1.0, atol=tol):
        errors.append("a customer is not fully assigned")
    if (x[~y] > tol).any():
        errors.append("a customer is assigned to a closed facility")
    if (facility_load(instance, sol) > instance.capacity + tol).any():
        errors.append("a facility capacity is exceeded")
    single = ~oversized_customers(instance)
    xs = x[:, single]
    if not (np.isclose(xs, 0, atol=tol) | np.isclose(xs, 1, atol=tol)).all():
        errors.append("a normal customer is split across facilities")
    return errors
