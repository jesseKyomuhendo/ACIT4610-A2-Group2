"""Representation and repair, shared by both MOEAs.

Chromosome: y[i] = 1 if facility i is open. decode() turns it into a feasible solution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cflp_moea.data_loader import CFLPInstance

_EPS = 1e-9


@dataclass(frozen=True)
class Solution:
    y: np.ndarray       # open facilities after repair
    x: np.ndarray       # x[i, j] = share of customer j served by facility i


def facility_load(instance: CFLPInstance, sol: Solution) -> np.ndarray:
    return sol.x @ instance.demand


def oversized_customers(instance: CFLPInstance) -> np.ndarray:
    """Customers too big for any facility (only in cap41/42)."""
    return instance.demand > instance.capacity.max()


def random_chromosome(instance: CFLPInstance, rng: np.random.Generator,
                      p_open: float = 0.5) -> np.ndarray:
    return rng.random(instance.m) < p_open


def decode(instance: CFLPInstance, chromosome: np.ndarray) -> Solution:
    m, n = instance.m, instance.n
    cap, F, d, C = instance.capacity, instance.fixed_cost, instance.demand, instance.alloc_cost

    y = np.asarray(chromosome, dtype=bool).copy()

    # open more facilities until total demand fits
    for i in np.argsort(F / cap, kind="stable"):
        if cap[y].sum() >= d.sum():
            break
        y[i] = True

    residual = np.where(y, cap, 0.0).astype(float)
    x = np.zeros((m, n))
    big = oversized_customers(instance)

    # assign largest customers first, to the cheapest open facility with room
    for j in np.argsort(-d, kind="stable"):
        if big[j]:
            _split_customer(j, y, residual, x, instance)
            continue
        fits = y & (residual >= d[j] - _EPS)
        if fits.any():
            i = _argmin_where(C[:, j], fits)
        else:
            # no room: open the cheapest closed facility that fits
            can_open = ~y & (cap >= d[j] - _EPS)
            if not can_open.any():
                raise RuntimeError(f"{instance.name}: no facility can take customer {j}")
            i = _argmin_where(F + C[:, j], can_open)
            y[i] = True
            residual[i] = cap[i]
        x[i, j] = 1.0
        residual[i] -= d[j]

    # close facilities that serve nobody
    y &= x.sum(axis=1) > 0
    return Solution(y=y, x=x)


def _split_customer(j: int, y: np.ndarray, residual: np.ndarray, x: np.ndarray,
                    instance: CFLPInstance) -> None:
    """Split an oversized customer over the cheapest facilities."""
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
    """Return the broken constraints (empty list = feasible)."""
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
