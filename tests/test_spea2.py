import numpy as np

from cflp_moea.algorithms import spea2
from cflp_moea.algorithms.common import non_dominated_mask
from cflp_moea.data_loader import load_instance
from cflp_moea.objectives import objectives
from cflp_moea.representation import check_feasible, decode
from cflp_moea.utils import make_rng


def test_fitness_raw_part():
    # 0 and 1 non-dominated; 0 dominates 2; 0, 1 and 2 dominate 3
    F = np.array([[1.0, 3.0], [3.0, 1.0], [2.0, 4.0], [4.0, 4.0]])
    fit = spea2.fitness(F)
    raw = np.floor(fit)
    assert raw.tolist() == [0, 0, 2, 4]           # S(0)=2, S(1)=1, S(2)=1
    assert (fit - raw < 1).all()


def test_truncate_removes_most_crowded():
    F = np.array([[0.0, 10.0], [5.0, 5.0], [5.1, 4.9], [10.0, 0.0]])
    keep = spea2.truncate(F, 3)
    assert 0 in keep and 3 in keep and len(keep) == 3


def test_environmental_selection_fills_with_dominated():
    F = np.array([[1.0, 3.0], [3.0, 1.0], [2.0, 4.0], [4.0, 4.0]])
    chosen, _ = spea2.environmental_selection(F, 3)
    assert sorted(chosen.tolist()) == [0, 1, 2]


def test_run_on_real_data():
    settings = dict(pop_size=30, max_evaluations=1500, crossover_prob=0.9, mutation_prob=0.05)
    for name in ["cap41", "cap101", "cap121"]:
        inst = load_instance(name)
        res = spea2.run(inst, make_rng(4610), **settings)
        assert res.n_evals == 1500
        assert len(res.F) >= 1 and non_dominated_mask(res.F).all()
        for x, f in zip(res.X, res.F):
            sol = decode(inst, x)
            assert check_feasible(inst, sol) == []
            assert np.allclose(objectives(inst, sol), f)


def test_same_seed_same_result():
    inst = load_instance("cap101")
    settings = dict(pop_size=20, max_evaluations=400, crossover_prob=0.9, mutation_prob=0.05)
    a = spea2.run(inst, make_rng(1), **settings)
    b = spea2.run(inst, make_rng(1), **settings)
    assert np.array_equal(a.F, b.F)
