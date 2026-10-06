"""SPEA2 (Zitzler, Laumanns & Thiele, 2001), implemented by hand.

Main loop (one generation):
  1. Fitness assignment on the union of population and archive (P + A):
       strength S(i) = number of solutions i dominates
       raw R(i)      = sum of S(j) over all j that dominate i   (0 = non-dominated)
       density D(i)  = 1 / (sigma_k(i) + 2), sigma_k = distance to the k-th
                       nearest neighbour in objective space
       fitness F(i)  = R(i) + D(i)        (lower is better; F < 1 = non-dominated)
  2. Environmental selection -> new archive of fixed size:
       copy all non-dominated solutions (F < 1);
       too few  -> fill up with the best dominated solutions (lowest F);
       too many -> truncation: repeatedly remove the solution closest to its
                   nearest neighbour (ties: compare 2nd nearest, then 3rd, ...).
  3. Termination: stop when another generation would exceed max_evaluations.
  4. Mating selection: binary tournament on fitness, from the ARCHIVE only.
  5. Variation: crossover + mutation (src/representation.py), then repair
     (src/repair.py) and evaluation (src/objectives.py) -> new population.

Representation, variation and repair are shared with NSGA-II (src/nsga2.py),
so only the selection and replacement steps differ between the two algorithms.

Distances are computed on objectives scaled to [0, 1] with the min and max of
the current union, because f1 and f2 have very different magnitudes.
"""

import numpy as np

from src.objectives import dominance_matrix, evaluate, non_dominated_indices
from src.repair import repair
from src.representation import crossover, initial_population, mutate
from src.util.fetch_config import SPEA2_K, TOURNAMENT_SIZE


# -----------------------------------------------------------------------------
# Distances in objective space
# -----------------------------------------------------------------------------
def pairwise_distances(objectives):
    """Euclidean distances between all solutions, objectives scaled to [0, 1].

    The distance of a solution to itself is set to infinity, so it is never
    counted as its own neighbour.
    """
    low, high = objectives.min(axis=0), objectives.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    scaled = (objectives - low) / span
    diff = scaled[:, None, :] - scaled[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    return dist


# -----------------------------------------------------------------------------
# Fitness assignment
# -----------------------------------------------------------------------------
def spea2_fitness(objectives, k):
    """Return F = raw fitness + density for every solution (lower is better)."""
    D = dominance_matrix(objectives)                  # D[p, q]: p dominates q
    strength = D.sum(axis=1)                          # S(p): how many p dominates
    raw = (D * strength[:, None]).sum(axis=0)         # R(q): sum of S(p) over p dominating q

    dist = pairwise_distances(objectives)
    sigma_k = np.sort(dist, axis=1)[:, k - 1]         # distance to the k-th nearest neighbour
    density = 1.0 / (sigma_k + 2.0)                   # always < 1
    return raw + density


# -----------------------------------------------------------------------------
# Environmental selection (archive update)
# -----------------------------------------------------------------------------
def truncate(objectives, candidates, archive_size):
    """Remove solutions from `candidates` until archive_size are left.

    Each step removes the solution with the smallest distance to its nearest
    neighbour; ties are broken by the 2nd nearest distance, then the 3rd, ...
    This keeps boundary solutions and spreads the archive along the front.
    """
    dist = pairwise_distances(objectives[candidates])
    alive = np.ones(len(candidates), dtype=bool)

    while alive.sum() > archive_size:
        nearest = np.where(alive, dist.min(axis=1), np.inf)
        tied = np.flatnonzero(nearest == nearest.min())
        if len(tied) == 1:
            remove = tied[0]
        else:
            # sorted distance lists of the tied solutions; remove the
            # lexicographically smallest (np.lexsort uses the LAST key first)
            sorted_rows = np.sort(dist[tied], axis=1)
            remove = tied[np.lexsort(sorted_rows.T[::-1])[0]]
        alive[remove] = False
        dist[remove, :] = np.inf                       # removed solutions are
        dist[:, remove] = np.inf                       # no longer neighbours

    return candidates[alive]


def environmental_selection(objectives, fitness, archive_size):
    """Return the indices (into the union) of the next archive."""
    non_dominated = np.flatnonzero(fitness < 1)
    if len(non_dominated) == archive_size:
        return non_dominated
    if len(non_dominated) < archive_size:
        best_first = np.argsort(fitness)              # non-dominated first, then best dominated
        return best_first[:archive_size]
    return truncate(objectives, non_dominated, archive_size)


# -----------------------------------------------------------------------------
# Mating selection
# -----------------------------------------------------------------------------
def tournament(fitness, rng):
    """Pick one archive index by tournament on fitness (lower wins)."""
    candidates = rng.choice(len(fitness), size=TOURNAMENT_SIZE, replace=False)
    return min(candidates, key=lambda c: fitness[c])


def make_offspring(archive, archive_fitness, inst, params, rng):
    """Create pop_size repaired offspring from parents chosen in the archive."""
    offspring = []
    while len(offspring) < params["pop_size"]:
        parent1 = archive[tournament(archive_fitness, rng)]
        parent2 = archive[tournament(archive_fitness, rng)]
        child1, child2 = crossover(parent1, parent2, params["crossover_prob"], rng)
        for child in (child1, child2):
            child = mutate(child, inst, params["mutation_prob"], rng)
            offspring.append(repair(child, inst))
    return offspring[:params["pop_size"]]


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def run_spea2(inst, params, rng):
    """Run SPEA2 on one instance with one parameter configuration.

    params: one entry of CONFIGURATIONS in config.yaml
    rng:    numpy random generator (seeded by the caller)

    Returns a dict (same keys as run_nsga2):
      front        objective vectors [f1, f2] of the final non-dominated set
      solutions    the matching chromosomes
      evaluations  number of objective evaluations used
      generations  number of generations completed
    """
    pop_size, archive_size = params["pop_size"], params["archive_size"]
    max_evaluations = params["max_evaluations"]
    k = SPEA2_K or int(np.sqrt(pop_size + archive_size))

    population = [repair(c, inst) for c in initial_population(inst, pop_size, rng)]
    objectives = np.array([evaluate(c, inst) for c in population])
    evaluations, generations = pop_size, 0
    archive, archive_objectives = [], np.empty((0, 2))   # the archive starts empty

    while True:
        # 1-2. fitness on P + A, then the new archive
        union = population + archive
        union_objectives = np.vstack([objectives, archive_objectives])
        fitness = spea2_fitness(union_objectives, k)
        chosen = environmental_selection(union_objectives, fitness, archive_size)
        archive = [union[c] for c in chosen]
        archive_objectives = union_objectives[chosen]
        archive_fitness = fitness[chosen]

        # 3. termination
        if evaluations + pop_size > max_evaluations:
            break

        # 4-5. parents from the archive -> new population
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