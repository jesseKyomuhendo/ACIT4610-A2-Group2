"""Feasibility repair (capacity constraint).

The representation in src/representation.py already guarantees two of the
three constraints (every customer assigned exactly once, only to an open
facility). Crossover and mutation can still overload a facility, i.e. assign
more demand to it than its capacity. repair() fixes that.

Repair procedure
----------------
Repeat until no facility is overloaded:
  1. Take the most overloaded facility i.
  2. For every customer j currently on i, and every other OPEN facility t that
     has enough spare capacity for j's demand, compute the cost change of
     moving j from i to t:  C[t, j] - C[i, j]   (change in f2)
  3. Make the cheapest move. If facility i becomes empty it is closed.
  4. Only if no open facility has room for any of i's customers, closed
     facilities are also considered; then the cost change also includes the
     opening cost F[t] (change in f1), and the cheapest facility is opened.

Repair makes the smallest change needed: it never opens a facility unless it
has to. Deciding the trade-off between f1 and f2 is left to the MOEAs.

Why it always stops: a customer is only moved to a facility that still has
room for it afterwards, so no new overloads are created and each move reduces
the overload. Each customer is moved at most once, so at most n moves are made.

The repair is deterministic (no randomness), so the same chromosome always
gives the same feasible solution. The repaired chromosome replaces the
original in the population (Lamarckian repair), in both NSGA-II and SPEA2.
"""

import numpy as np

from src.representation import facility_loads, open_facilities


def is_feasible(chromosome, inst):
    """True if no facility serves more demand than its capacity."""
    return bool(np.all(facility_loads(chromosome, inst) <= inst.capacity))


def repair(chromosome, inst, verbose=False):
    """Return a feasible copy of the chromosome (the input is not changed).

    Set verbose=True to print every move, e.g. for the worked example in the report.
    """
    chromosome = chromosome.copy()
    loads = facility_loads(chromosome, inst)

    while True:
        overload = loads - inst.capacity
        if np.all(overload <= 0):
            return chromosome

        i = int(np.argmax(overload))                     # most overloaded facility
        j, t, extra_cost = _cheapest_move(chromosome, i, loads, inst, allow_opening=False)
        if j is None:
            j, t, extra_cost = _cheapest_move(chromosome, i, loads, inst, allow_opening=True)
        if j is None:
            raise RuntimeError(f"{inst.name}: no customer on facility {i} can be moved "
                               f"to any facility with enough spare capacity.")

        if verbose:
            was_closed = loads[t] == 0
            print(f"  facility {i} overloaded ({loads[i]:,.0f} > {inst.capacity[i]:,.0f}): "
                  f"move customer {j} (demand {inst.demand[j]:,.0f}) to facility {t}"
                  f"{' (opened)' if was_closed else ''}, cost change {extra_cost:+,.2f}")

        chromosome[j] = t
        loads[i] -= inst.demand[j]
        loads[t] += inst.demand[j]


def _cheapest_move(chromosome, i, loads, inst, allow_opening):
    """Find the cheapest (customer, target facility) move away from facility i.

    allow_opening=False: only open facilities are possible targets.
    allow_opening=True:  closed facilities are allowed too (their opening cost is added).
    Returns (j, t, extra_cost), or (None, None, None) if no move is possible.
    """
    is_closed = open_facilities(chromosome, inst.m) == 0
    spare = inst.capacity - loads

    best_j, best_t, best_extra = None, None, np.inf
    for j in np.flatnonzero(chromosome == i):
        # cost change of moving customer j to each facility
        extra = inst.cost[:, j] - inst.cost[i, j] + inst.fixed_cost * is_closed
        allowed = spare >= inst.demand[j]                # enough room for customer j
        allowed[i] = False                               # must leave facility i
        if not allow_opening:
            allowed &= ~is_closed
        extra[~allowed] = np.inf

        t = int(np.argmin(extra))
        if extra[t] < best_extra:
            best_j, best_t, best_extra = int(j), t, float(extra[t])

    return best_j, best_t, best_extra