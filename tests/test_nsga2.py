import numpy as np

from cflp_moea.algorithms import nsga2
from cflp_moea.algorithms.common import non_dominated_mask
from cflp_moea.data_loader import load_instance
from cflp_moea.objectives import objectives
from cflp_moea.representation import check_feasible, decode
from cflp_moea.utils import make_rng


def test_fast_non_dominated_sort():
    F = np.array([[1, 5], [2, 2], [5, 1], [3, 3], [4, 4], [2, 2]])
    fronts = [sorted(f.tolist()) for f in nsga2.fast_non_dominated_sort(F)]
    assert fronts == [[0, 1, 2, 5], [3], [4]]


def test_crowding_distance():
    F = np.array([[0.0, 4.0], [1.0, 2.0], [4.0, 0.0]])
    cd = nsga2.crowding_distance(F)
    assert np.isinf(cd[0]) and np.isinf(cd[2])
    assert np.isclose(cd[1], 4 / 4 + 4 / 4)


def test_environmental_selection_keeps_best_fronts():
    F = np.array([[1, 5], [5, 1], [3, 3], [4, 4], [6, 6]])
    keep = nsga2.environmental_selection(F, 3)
    assert sorted(keep.tolist()) == [0, 1, 2]


def test_run_on_real_data():
    settings = dict(pop_size=30, max_evaluations=1500, crossover_prob=0.9, mutation_prob=0.05)
    for name in ["cap41", "cap101", "cap121"]:
        inst = load_instance(name)
        res = nsga2.run(inst, make_rng(4610), **settings)
        assert res.n_evals == 1500
        assert len(res.F) >= 1 and non_dominated_mask(res.F).all()
        for x, f in zip(res.X, res.F):
            sol = decode(inst, x)
            assert check_feasible(inst, sol) == []
            assert np.allclose(objectives(inst, sol), f)


def test_same_seed_same_result():
    inst = load_instance("cap101")
    settings = dict(pop_size=20, max_evaluations=400, crossover_prob=0.9, mutation_prob=0.05)
    a = nsga2.run(inst, make_rng(1), **settings)
    b = nsga2.run(inst, make_rng(1), **settings)
    assert np.array_equal(a.F, b.F)
