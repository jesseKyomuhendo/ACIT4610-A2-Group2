"""Chromosome representation, initialisation and variation operators shared by NSGA-II and SPEA2.

A chromosome has one gene per customer, and chromosome[j] = i means customer j is served by facility i.
"""

import numpy as np


# Decoding
# Every customer gets exactly one facility and a facility is open only when it serves someone,
# so only the capacity constraint can be broken and repair.py fixes that
def open_facilities(chromosome, m):
    """Returns y, where y[i] is 1 if facility i serves at least one customer and 0 otherwise."""
    y = np.zeros(m, dtype=int)
    y[chromosome] = 1
    return y


def facility_loads(chromosome, inst):
    """Returns the total demand assigned to each facility."""
    return np.bincount(chromosome, weights=inst.demand, minlength=inst.m)


# Initialisation
def random_individual(inst, rng):
    """Creates one random chromosome, which may break capacity until it is repaired."""
    # Picking the number of facilities first spreads the population from few to many open facilities
    # Fully random assignment would open almost every facility in every individual
    k = rng.integers(1, inst.m + 1)
    chosen = rng.choice(inst.m, size=k, replace=False)
    return rng.choice(chosen, size=inst.n)          # each customer gets one of the k facilities


def initial_population(inst, pop_size, rng):
    """Creates pop_size random chromosomes before repair."""
    return [random_individual(inst, rng) for _ in range(pop_size)]


# Crossover
def crossover(parent1, parent2, crossover_prob, rng):
    """Uniform crossover that returns two children."""
    # Without crossover the children are copies of the parents
    if rng.random() >= crossover_prob:
        return parent1.copy(), parent2.copy()

    # Each customer comes from either parent with equal chance and child2 gets the opposite choice
    take_from_parent1 = rng.random(len(parent1)) < 0.5
    child1 = np.where(take_from_parent1, parent1, parent2)
    child2 = np.where(take_from_parent1, parent2, parent1)
    return child1, child2


# Mutation
def mutate(chromosome, inst, mutation_prob, rng):
    """Reassigns each customer with probability mutation_prob and returns a new chromosome."""
    child = chromosome.copy()
    genes_to_change = np.flatnonzero(rng.random(inst.n) < mutation_prob)
    if len(genes_to_change) == 0:
        return child

    currently_open = np.flatnonzero(open_facilities(chromosome, inst.m))
    for j in genes_to_change:
        # An already open facility can empty and close another one, which lowers f1
        if rng.random() < 0.5:
            child[j] = rng.choice(currently_open)
        # Any facility can open a new one, which can lower f2
        else:
            child[j] = rng.integers(inst.m)
    return child