"""NSGA-II: fast non-dominated sorting, crowding distance,
crowded tournament selection, elitist (mu + lambda) environmental selection.
"""
from __future__ import annotations

import numpy as np

from cflp_moea.algorithms.common import (RunResult, binary_tournament, dominance_matrix,
                                         unique_non_dominated)
from cflp_moea.data_loader import CFLPInstance
from cflp_moea.objectives import Evaluator
from cflp_moea.operators import init_population, make_offspring


def fast_non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """Split solutions into fronts: front 0 is non-dominated, front 1 is next, etc."""
    D = dominance_matrix(F)
    dominated_by_me = [np.flatnonzero(row) for row in D]
    n_dominating_me = D.sum(axis=0)
    n = len(F)

    fronts = []
    current = [p for p in range(n) if n_dominating_me[p] == 0]
    while current:
        fronts.append(np.array(current))
        nxt = []
        for p in current:
            for q in dominated_by_me[p]:
                n_dominating_me[q] -= 1
                if n_dominating_me[q] == 0:
                    nxt.append(q)
        current = nxt
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Crowding distance within one front; boundary solutions get infinity."""
    n = len(F)
    dist = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(F.shape[1]):
        order = np.argsort(F[:, k], kind="stable")
        span = F[order[-1], k] - F[order[0], k]
        dist[order[0]] = dist[order[-1]] = np.inf
        if span == 0:
            continue
        dist[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / span
    return dist


def rank_and_crowding(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rank = np.empty(len(F), dtype=int)
    crowd = np.empty(len(F))
    for r, front in enumerate(fast_non_dominated_sort(F)):
        rank[front] = r
        crowd[front] = crowding_distance(F[front])
    return rank, crowd


def environmental_selection(F: np.ndarray, size: int) -> np.ndarray:
    """Keep the best `size`: whole fronts first, then the least crowded from the last front."""
    keep = []
    for front in fast_non_dominated_sort(F):
        if len(keep) + len(front) <= size:
            keep.extend(front)
        else:
            cd = crowding_distance(F[front])
            keep.extend(front[np.argsort(-cd, kind="stable")][: size - len(keep)])
            break
    return np.array(keep)


def run(instance: CFLPInstance, rng: np.random.Generator, pop_size: int, max_evaluations: int,
        crossover_prob: float, mutation_prob: float, init: str = "random_repair") -> RunResult:
    ev = Evaluator(instance)

    def evaluate(pop):
        out = [ev(c) for c in pop]
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])

    X, F = evaluate(init_population(instance, pop_size, rng, init))

    while ev.n_evals < max_evaluations:
        # crowded tournament: lower rank wins, then larger crowding distance
        rank, crowd = rank_and_crowding(F)
        score = np.empty(len(F))
        score[np.lexsort((-crowd, rank))] = np.arange(len(F))
        parents = X[binary_tournament(score, pop_size, rng)]

        n_children = min(pop_size, max_evaluations - ev.n_evals)
        children = make_offspring(parents, rng, crossover_prob, mutation_prob)[:n_children]
        Xc, Fc = evaluate(children)

        # elitism: parents + children compete for the next population
        X, F = np.vstack([X, Xc]), np.vstack([F, Fc])
        keep = environmental_selection(F, pop_size)
        X, F = X[keep], F[keep]

    Xn, Fn = unique_non_dominated(X, F)
    return RunResult(X=Xn, F=Fn, n_evals=ev.n_evals)
