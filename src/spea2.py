"""SPEA2 implemented by hand.

It shares representation, variation and repair with NSGA-II, so only selection and replacement differ.
"""

import numpy as np

from src.objectives import dominance_matrix, evaluate, non_dominated_indices
from src.repair import repair
from src.representation import crossover, initial_population, mutate
from src.util.fetch_config import SPEA2_K, TOURNAMENT_SIZE


# Distances in objective space
def pairwise_distances(objectives):
    """Euclidean distances between all solutions, with objectives scaled to [0, 1].

    Scaling is needed because f1 and f2 have very different magnitudes.
    """
    low, high = objectives.min(axis=0), objectives.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    scaled = (objectives - low) / span
    diff = scaled[:, None, :] - scaled[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    # A solution is never counted as its own neighbour
    np.fill_diagonal(dist, np.inf)
    return dist


# Fitness assignment
def spea2_fitness(objectives, k):
    """Returns fitness F as raw fitness plus density for every solution, where lower is better."""
    D = dominance_matrix(objectives)                  # D[p, q] is True when p dominates q
    strength = D.sum(axis=1)                          # S(p) is how many solutions p dominates
    raw = (D * strength[:, None]).sum(axis=0)         # R(q) sums S(p) over every p dominating q

    # Density breaks ties between solutions with the same raw fitness
    dist = pairwise_distances(objectives)
    sigma_k = np.sort(dist, axis=1)[:, k - 1]         # distance to the k-th nearest neighbour
    density = 1.0 / (sigma_k + 2.0)                   # always below 1, so F < 1 means non-dominated
    return raw + density


# Environmental selection that updates the archive
def truncate(objectives, candidates, archive_size):
    """Removes solutions from candidates until archive_size are left.

    This keeps boundary solutions and spreads the archive along the front.
    """
    dist = pairwise_distances(objectives[candidates])
    alive = np.ones(len(candidates), dtype=bool)

    # Each step removes the solution closest to its nearest neighbour
    while alive.sum() > archive_size:
        nearest = np.where(alive, dist.min(axis=1), np.inf)
        tied = np.flatnonzero(nearest == nearest.min())
        if len(tied) == 1:
            remove = tied[0]
        else:
            # Ties are broken by the 2nd nearest distance, then the 3rd and so on
            # np.lexsort uses the last key first, so the columns are reversed
            sorted_rows = np.sort(dist[tied], axis=1)
            remove = tied[np.lexsort(sorted_rows.T[::-1])[0]]
        alive[remove] = False
        dist[remove, :] = np.inf                       # removed solutions are
        dist[:, remove] = np.inf                       # no longer neighbours

    return candidates[alive]


def environmental_selection(objectives, fitness, archive_size):
    """Returns the indices in the union of the solutions that form the next archive."""
    non_dominated = np.flatnonzero(fitness < 1)
    if len(non_dominated) == archive_size:
        return non_dominated
    # Too few non-dominated solutions, so fill up with the best dominated ones
    if len(non_dominated) < archive_size:
        best_first = np.argsort(fitness)              # non-dominated first, then best dominated
        return best_first[:archive_size]
    # Too many non-dominated solutions, so truncate
    return truncate(objectives, non_dominated, archive_size)


# Mating selection
def tournament(fitness, rng):
    """Picks one archive index by tournament on fitness, where lower wins."""
    candidates = rng.choice(len(fitness), size=TOURNAMENT_SIZE, replace=False)
    return min(candidates, key=lambda c: fitness[c])


def make_offspring(archive, archive_fitness, inst, params, rng):
    """Creates pop_size repaired offspring from parents chosen in the archive only."""
    offspring = []
    while len(offspring) < params["pop_size"]:
        parent1 = archive[tournament(archive_fitness, rng)]
        parent2 = archive[tournament(archive_fitness, rng)]
        child1, child2 = crossover(parent1, parent2, params["crossover_prob"], rng)
        for child in (child1, child2):
            child = mutate(child, inst, params["mutation_prob"], rng)
            offspring.append(repair(child, inst))
    return offspring[:params["pop_size"]]


# Main loop
def run_spea2(inst, params, rng):
    """Runs SPEA2 on one instance with one configuration from config.yaml and a seeded rng.

    Returns the same result keys as run_nsga2.
    """
    pop_size, archive_size = params["pop_size"], params["archive_size"]
    max_evaluations = params["max_evaluations"]
    k = SPEA2_K or int(np.sqrt(pop_size + archive_size))   # default k from the original paper

    population = [repair(c, inst) for c in initial_population(inst, pop_size, rng)]
    objectives = np.array([evaluate(c, inst) for c in population])
    evaluations, generations = pop_size, 0
    archive, archive_objectives = [], np.empty((0, 2))   # the archive starts empty

    while True:
        # Fitness on population and archive together, then the new archive
        union = population + archive
        union_objectives = np.vstack([objectives, archive_objectives])
        fitness = spea2_fitness(union_objectives, k)
        chosen = environmental_selection(union_objectives, fitness, archive_size)
        archive = [union[c] for c in chosen]
        archive_objectives = union_objectives[chosen]
        archive_fitness = fitness[chosen]

        # Stop when another generation would exceed the evaluation budget
        if evaluations + pop_size > max_evaluations:
            break

        # Parents from the archive create the new population
        population = make_offspring(archive, archive_fitness, inst, params, rng)
        objectives = np.array([evaluate(c, inst) for c in population])
        evaluations += pop_size
        generations += 1

    best = non_dominated_indices(archive_objectives)
    return {
        "front": archive_objectives[best],
        "solutions": [archive[c] for c in best],
        "evaluations": evaluations,
        "generations": generations,
    }