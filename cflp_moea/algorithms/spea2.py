"""SPEA2: strength/raw fitness, k-th nearest neighbour density, external archive,
archive truncation, binary tournament selection on the archive.
"""
from __future__ import annotations

import numpy as np

from cflp_moea.algorithms.common import (RunResult, binary_tournament, dominance_matrix,
                                         unique_non_dominated)
from cflp_moea.data_loader import CFLPInstance
from cflp_moea.objectives import Evaluator
from cflp_moea.operators import init_population, make_offspring


def _distances(F: np.ndarray) -> np.ndarray:
    """Distances in objective space after scaling each objective to [0, 1]."""
    span = F.max(axis=0) - F.min(axis=0)
    Z = (F - F.min(axis=0)) / np.where(span > 0, span, 1.0)
    dist = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    return dist


def fitness(F: np.ndarray) -> np.ndarray:
    """F(i) = R(i) + D(i). Below 1 means non-dominated."""
    D = dominance_matrix(F)
    strength = D.sum(axis=1)                      # S(i): how many i dominates
    raw = (D * strength[:, None]).sum(axis=0)     # R(i): sum of S(j) over j dominating i
    k = int(np.sqrt(len(F)))
    sigma_k = np.sort(_distances(F), axis=1)[:, k - 1]
    density = 1.0 / (sigma_k + 2.0)               # D(i), always < 1
    return raw + density


def truncate(F: np.ndarray, size: int) -> np.ndarray:
    """Remove the most crowded solution one by one until `size` remain."""
    keep = np.arange(len(F))
    dist = _distances(F)
    while len(keep) > size:
        d = np.sort(dist[np.ix_(keep, keep)], axis=1)
        # smallest distance to nearest neighbour, ties broken by the next neighbours
        worst = np.lexsort(d.T[::-1])[0]
        keep = np.delete(keep, worst)
    return keep


def environmental_selection(F: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Next archive: all non-dominated, then filled with the best fitness or truncated."""
    fit = fitness(F)
    nd = np.flatnonzero(fit < 1)
    if len(nd) > size:
        chosen = nd[truncate(F[nd], size)]
    else:
        chosen = np.argsort(fit, kind="stable")[:size]
    return chosen, fit[chosen]


def run(instance: CFLPInstance, rng: np.random.Generator, pop_size: int, max_evaluations: int,
        crossover_prob: float, mutation_prob: float, init: str = "random_repair") -> RunResult:
    ev = Evaluator(instance)
    archive_size = pop_size

    def evaluate(pop):
        out = [ev(c) for c in pop]
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])

    X, F = evaluate(init_population(instance, pop_size, rng, init))
    AX, AF = X[:0], F[:0]                         # archive starts empty

    while True:
        # population + archive -> new archive
        UX, UF = np.vstack([X, AX]), np.vstack([F, AF])
        chosen, afit = environmental_selection(UF, archive_size)
        AX, AF = UX[chosen], UF[chosen]
        if ev.n_evals >= max_evaluations:
            break

        # mating: binary tournament on the archive, lower fitness wins
        parents = AX[binary_tournament(afit, pop_size, rng)]
        n_children = min(pop_size, max_evaluations - ev.n_evals)
        X, F = evaluate(make_offspring(parents, rng, crossover_prob, mutation_prob)[:n_children])

    Xn, Fn = unique_non_dominated(AX, AF)
    return RunResult(X=Xn, F=Fn, n_evals=ev.n_evals)
