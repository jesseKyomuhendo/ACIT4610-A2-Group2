import numpy as np

from cflp_moea.data_loader import load_instance
from cflp_moea.objectives import Evaluator
from cflp_moea.operators import (bit_flip_mutation, init_population, make_offspring,
                                 uniform_crossover)
from cflp_moea.utils import Timer, make_rng, run_seed


def test_init_population_shape_and_spread():
    inst = load_instance("cap121")
    pop = init_population(inst, 200, make_rng(4610))
    assert pop.shape == (200, inst.m) and pop.dtype == bool
    n_open = pop.sum(axis=1)
    assert n_open.min() < 10 and n_open.max() > 40


def test_same_seed_gives_same_population():
    inst = load_instance("cap41")
    a = init_population(inst, 50, make_rng(run_seed(4610, 3)))
    b = init_population(inst, 50, make_rng(run_seed(4610, 3)))
    c = init_population(inst, 50, make_rng(run_seed(4610, 4)))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_uniform_crossover_keeps_genes_and_respects_prob():
    rng = make_rng(0)
    a, b = np.zeros(50, dtype=bool), np.ones(50, dtype=bool)
    c1, c2 = uniform_crossover(a, b, rng, prob=1.0)
    assert np.array_equal(c1, ~c2)
    assert 0 < c1.sum() < 50
    c1, c2 = uniform_crossover(a, b, rng, prob=0.0)
    assert np.array_equal(c1, a) and np.array_equal(c2, b)


def test_bit_flip_mutation_rate():
    rng = make_rng(0)
    y = np.zeros(10_000, dtype=bool)
    assert abs(bit_flip_mutation(y, rng, 0.05).mean() - 0.05) < 0.01
    assert not bit_flip_mutation(y, rng, 0.0).any()


def test_offspring_decode_feasible_on_real_data():
    for name in ["cap41", "cap101", "cap121"]:
        inst = load_instance(name)
        rng = make_rng(1)
        ev = Evaluator(inst)
        parents = init_population(inst, 51, rng)
        children = make_offspring(parents, rng, 0.9, 0.05)
        assert children.shape == parents.shape
        for c in children:
            y, f = ev(c)
            assert y.any() and np.isfinite(f).all()


def test_timer():
    with Timer() as t:
        sum(range(1000))
    assert t.seconds >= 0
