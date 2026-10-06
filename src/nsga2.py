"""NSGA-II (Deb et al., 2002), implemented by hand.

Main loop (one generation):
  1. Mating selection: binary tournament with the crowded-comparison operator
     (lower front rank wins; equal rank -> larger crowding distance wins).
  2. Variation: crossover + mutation (src/representation.py), then repair
     (src/repair.py) and evaluation (src/objectives.py) -> N offspring.
  3. Environmental selection (elitist): combine parents and offspring (2N),
     sort them into non-dominated fronts, and fill the next population front
     by front. The last front that does not fit completely is cut using
     crowding distance (least crowded solutions are kept).
Termination: stop when another generation would exceed max_evaluations.

Representation, variation and repair are shared with SPEA2 (src/spea2.py), so
only the selection and replacement steps differ between the two algorithms.
"""

import numpy as np

from src.objectives import dominance_matrix, evaluate, non_dominated_indices
from src.repair import repair
from src.representation import crossover, initial_population, mutate
from src.util.fetch_config import TOURNAMENT_SIZE


# -----------------------------------------------------------------------------
# Non-dominated sorting
# -----------------------------------------------------------------------------
def fast_non_dominated_sort(objectives):
    """Sort solutions into fronts.

    Returns (fronts, rank):
      fronts: list of index arrays; fronts[0] is the best (non-dominated) front
      rank:   rank[p] = number of the front solution p belongs to (0 = best)
    """
    D = dominance_matrix(objectives)        # D[p, q]: p dominates q
    domination_count = D.sum(axis=0)        # how many solutions dominate q
    rank = np.zeros(len(objectives), dtype=int)

    fronts = []
    current = np.flatnonzero(domination_count == 0)
    while len(current) > 0:
        fronts.append(current)
        rank[current] = len(fronts) - 1
        next_front = []
        for p in current:
            for q in np.flatnonzero(D[p]):  # every solution p dominates
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        current = np.array(next_front, dtype=int)
    return fronts, rank


# -----------------------------------------------------------------------------
# Crowding distance
# -----------------------------------------------------------------------------
def crowding_distance(objectives, front):
    """Crowding distance of every solution in one front (Lecture 4).

    For each objective, sort the front; the two boundary solutions get an
    infinite distance, every other solution gets the normalised distance
    between its two neighbours added. Larger = less crowded = preferred.
    Returns an array aligned with `front`.
    """
    values = objectives[front]
    distance = np.zeros(len(front))
    if len(front) <= 2:
        distance[:] = np.inf
        return distance

    for k in range(values.shape[1]):                 # each objective
        order = np.argsort(values[:, k])
        f_min, f_max = values[order[0], k], values[order[-1], k]
        distance[order[0]] = distance[order[-1]] = np.inf
        if f_max == f_min:
            continue
        for pos in range(1, len(front) - 1):
            neighbour_gap = values[order[pos + 1], k] - values[order[pos - 1], k]
            distance[order[pos]] += neighbour_gap / (f_max - f_min)
    return distance


# -----------------------------------------------------------------------------
# Selection
# -----------------------------------------------------------------------------
def crowded_tournament(rank, crowding, rng):
    """Pick one parent index by tournament with the crowded-comparison operator."""
    candidates = rng.choice(len(rank), size=TOURNAMENT_SIZE, replace=False)
    # best = lowest rank; among equal ranks, the largest crowding distance
    return min(candidates, key=lambda c: (rank[c], -crowding[c]))


def rank_and_crowding(objectives):
    """Front rank and crowding distance of every solution (used by the tournament)."""
    fronts, rank = fast_non_dominated_sort(objectives)
    crowding = np.zeros(len(objectives))
    for front in fronts:
        crowding[front] = crowding_distance(objectives, front)
    return rank, crowding


def environmental_selection(objectives, pop_size):
    """Choose pop_size survivors from the combined population (elitist).

    Fronts are added in order of rank; the first front that does not fit
    completely is cut by keeping its least crowded solutions.
    Returns the indices of the survivors.
    """
    fronts, _ = fast_non_dominated_sort(objectives)
    survivors = []
    for front in fronts:
        space_left = pop_size - len(survivors)
        if len(front) <= space_left:
            survivors.extend(front)                           # whole front fits
        else:
            crowding = crowding_distance(objectives, front)
            least_crowded = np.argsort(-crowding)[:space_left]
            survivors.extend(front[least_crowded])
            break
    return np.array(survivors)


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def make_offspring(population, rank, crowding, inst, params, rng):
    """Create pop_size repaired offspring from the current population."""
    offspring = []
    while len(offspring) < params["pop_size"]:
        parent1 = population[crowded_tournament(rank, crowding, rng)]
        parent2 = population[crowded_tournament(rank, crowding, rng)]
        child1, child2 = crossover(parent1, parent2, params["crossover_prob"], rng)
        for child in (child1, child2):
            child = mutate(child, inst, params["mutation_prob"], rng)
            offspring.append(repair(child, inst))
    return offspring[:params["pop_size"]]


def run_nsga2(inst, params, rng):
    """Run NSGA-II on one instance with one parameter configuration.

    params: one entry of CONFIGURATIONS in config.yaml
    rng:    numpy random generator (seeded by the caller)

    Returns a dict:
      front        objective vectors [f1, f2] of the final non-dominated set
      solutions    the matching chromosomes
      evaluations  number of objective evaluations used
      generations  number of generations completed
    """
    pop_size, max_evaluations = params["pop_size"], params["max_evaluations"]

    population = [repair(c, inst) for c in initial_population(inst, pop_size, rng)]
    objectives = np.array([evaluate(c, inst) for c in population])
    evaluations, generations = pop_size, 0

    rank, crowding = rank_and_crowding(objectives)

    while evaluations + pop_size <= max_evaluations:
        offspring = make_offspring(population, rank, crowding, inst, params, rng)
        offspring_objectives = np.array([evaluate(c, inst) for c in offspring])
        evaluations += len(offspring)

        combined = population + offspring
        combined_objectives = np.vstack([objectives, offspring_objectives])
        survivors = environmental_selection(combined_objectives, pop_size)

        population = [combined[k] for k in survivors]
        objectives = combined_objectives[survivors]
        rank, crowding = rank_and_crowding(objectives)
        generations += 1

    best = non_dominated_indices(objectives)
    return {
        "front": objectives[best],
        "solutions": [population[k] for k in best],
        "evaluations": evaluations,
        "generations": generations,
    }