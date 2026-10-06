"""Chromosome representation, initialization and variation operators.

Representation (shared by NSGA-II and SPEA2)
--------------------------------------------
A chromosome is an integer vector of length n (one gene per customer):

    chromosome[j] = i   means customer j is served by facility i

Example with 3 facilities (0, 1, 2) and 5 customers:

    chromosome = [2, 0, 2, 2, 0]
    -> customers 0, 2, 3 are served by facility 2, customers 1, 4 by facility 0
    -> open facilities y = [1, 0, 1]   (facility 1 serves nobody, so it is closed)

Decoding (genotype -> phenotype) is therefore one-to-one:
  - x_ij = 1 exactly when chromosome[j] == i, so every customer is assigned
    exactly once by construction;
  - y_i = 1 exactly when facility i serves at least one customer, so customers
    are only assigned to open facilities by construction.
Only the capacity constraint can be broken by the operators below;
src/repair.py fixes that.

All functions take a numpy random generator `rng`, so runs are reproducible.
"""

import numpy as np


# -----------------------------------------------------------------------------
# Decoding
# -----------------------------------------------------------------------------
def open_facilities(chromosome, m):
    """Return y: y[i] = 1 if facility i serves at least one customer, else 0."""
    y = np.zeros(m, dtype=int)
    y[chromosome] = 1
    return y


def facility_loads(chromosome, inst):
    """Return the total demand assigned to each facility, shape (m,)."""
    return np.bincount(chromosome, weights=inst.demand, minlength=inst.m)


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
def random_individual(inst, rng):
    """Create one random chromosome.

    1. Pick how many facilities to use, k, uniformly from 1..m.
    2. Pick k random facilities.
    3. Assign every customer to a random one of those k facilities.

    Choosing k first spreads the initial population from "few facilities open"
    (low f1) to "many open" (low f2). Assigning every customer to a uniformly
    random facility instead would open almost all facilities in every individual.
    The result may break capacity; repair is applied afterwards.
    """
    k = rng.integers(1, inst.m + 1)
    chosen = rng.choice(inst.m, size=k, replace=False)
    return rng.choice(chosen, size=inst.n)


def initial_population(inst, pop_size, rng):
    """Create pop_size random chromosomes (before repair)."""
    return [random_individual(inst, rng) for _ in range(pop_size)]


# -----------------------------------------------------------------------------
# Crossover
# -----------------------------------------------------------------------------
def crossover(parent1, parent2, crossover_prob, rng):
    """Uniform crossover. Returns two children.

    With probability crossover_prob, each gene (customer) is taken from parent1
    or parent2 with equal chance, and child2 gets the opposite choice.
    Otherwise the children are copies of the parents.
    """
    if rng.random() >= crossover_prob:
        return parent1.copy(), parent2.copy()

    take_from_parent1 = rng.random(len(parent1)) < 0.5
    child1 = np.where(take_from_parent1, parent1, parent2)
    child2 = np.where(take_from_parent1, parent2, parent1)
    return child1, child2


# -----------------------------------------------------------------------------
# Mutation
# -----------------------------------------------------------------------------
def mutate(chromosome, inst, mutation_prob, rng):
    """Reassign each customer to another facility with probability mutation_prob.

    The new facility is, with equal chance:
      - one that is already open in this chromosome (moves customers between
        open facilities, which can empty a facility and close it -> lower f1), or
      - any facility (can open a new facility -> lower f2).
    Returns a new chromosome; the input is not changed.
    """
    child = chromosome.copy()
    genes_to_change = np.flatnonzero(rng.random(inst.n) < mutation_prob)
    if len(genes_to_change) == 0:
        return child

    currently_open = np.flatnonzero(open_facilities(chromosome, inst.m))
    for j in genes_to_change:
        if rng.random() < 0.5:
            child[j] = rng.choice(currently_open)
        else:
            child[j] = rng.integers(inst.m)
    return child