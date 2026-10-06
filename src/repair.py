"""Feasibility repair for the capacity constraint, shared by NSGA-II and SPEA2.

The representation already guarantees the other two constraints, so only overloaded facilities need fixing.
"""

import numpy as np

from src.representation import facility_loads, open_facilities


def is_feasible(chromosome, inst):
    """True if no facility serves more demand than its capacity."""
    return bool(np.all(facility_loads(chromosome, inst) <= inst.capacity))


def repair(chromosome, inst, verbose=False):
    """Returns a feasible copy of the chromosome and leaves the input unchanged.

    Set verbose=True to print every move, for example for the worked example in the report.
    """
    # Repair has no randomness, so the same chromosome always gives the same solution
    chromosome = chromosome.copy()
    loads = facility_loads(chromosome, inst)

    # Each move only goes to a facility with room, so the loop always ends
    while True:
        overload = loads - inst.capacity
        if np.all(overload <= 0):
            return chromosome

        i = int(np.argmax(overload))                     # most overloaded facility
        # Prefer open facilities and only open a new one when no open facility has room
        # This keeps the change small and leaves the f1 and f2 trade-off to the MOEAs
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

        # Move the customer, and facility i closes by itself if it becomes empty
        chromosome[j] = t
        loads[i] -= inst.demand[j]
        loads[t] += inst.demand[j]


def _cheapest_move(chromosome, i, loads, inst, allow_opening):
    """Finds the cheapest move of one customer away from facility i.

    Returns customer, target facility and cost change, or three None values if no move is possible.
    """
    is_closed = open_facilities(chromosome, inst.m) == 0
    spare = inst.capacity - loads

    best_j, best_t, best_extra = None, None, np.inf
    for j in np.flatnonzero(chromosome == i):
        # Cost change of moving customer j to each facility, plus opening cost for closed ones
        extra = inst.cost[:, j] - inst.cost[i, j] + inst.fixed_cost * is_closed
        allowed = spare >= inst.demand[j]                # enough room for customer j
        allowed[i] = False                               # must leave facility i
        if not allow_opening:
            allowed &= ~is_closed                        # only open facilities are targets
        extra[~allowed] = np.inf

        t = int(np.argmin(extra))
        if extra[t] < best_extra:
            best_j, best_t, best_extra = int(j), t, float(extra[t])

    return best_j, best_t, best_extra