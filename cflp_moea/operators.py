"""Initialisation, crossover and mutation, shared by both MOEAs (Evaluator repairs children)."""
from __future__ import annotations

import numpy as np

from cflp_moea.data_loader import CFLPInstance


def init_population(instance: CFLPInstance, pop_size: int, rng: np.random.Generator,
                    method: str = "random_repair") -> np.ndarray:
    # each individual gets its own open probability, so we get few and many open facilities
    if method != "random_repair":
        raise ValueError(f"unknown initialisation method: {method}")
    p_open = rng.random((pop_size, 1))
    return rng.random((pop_size, instance.m)) < p_open


def uniform_crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator,
                      prob: float) -> tuple[np.ndarray, np.ndarray]:
    child_a, child_b = parent_a.copy(), parent_b.copy()
    if rng.random() < prob:
        swap = rng.random(parent_a.size) < 0.5
        child_a[swap], child_b[swap] = parent_b[swap], parent_a[swap]
    return child_a, child_b


def bit_flip_mutation(chromosome: np.ndarray, rng: np.random.Generator,
                      prob: float) -> np.ndarray:
    flip = rng.random(chromosome.size) < prob
    return chromosome ^ flip


def make_offspring(parents: np.ndarray, rng: np.random.Generator, crossover_prob: float,
                   mutation_prob: float) -> np.ndarray:
    n = len(parents)
    children = np.empty_like(parents)
    for k in range(0, n, 2):
        a = parents[k]
        b = parents[k + 1] if k + 1 < n else parents[0]
        c1, c2 = uniform_crossover(a, b, rng, crossover_prob)
        children[k] = bit_flip_mutation(c1, rng, mutation_prob)
        if k + 1 < n:
            children[k + 1] = bit_flip_mutation(c2, rng, mutation_prob)
    return children
