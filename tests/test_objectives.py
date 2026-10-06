import numpy as np

from cflp_moea.data_loader import CFLPInstance, load_instance, load_optimal_values
from cflp_moea.objectives import Evaluator, objectives
from cflp_moea.representation import Solution, decode


def tiny_instance():
    # 2 facilities, 3 customers; numbers chosen so results are easy to check by hand.
    return CFLPInstance(
        name="tiny", m=2, n=3,
        capacity=np.array([10.0, 10.0]),
        fixed_cost=np.array([100.0, 40.0]),
        demand=np.array([4.0, 3.0, 5.0]),
        alloc_cost=np.array([[1.0, 2.0, 3.0],
                             [10.0, 20.0, 30.0]]),
    )


def test_hand_checked_example():
    inst = tiny_instance()
    # Customers 0 and 1 to facility 0, customer 2 to facility 1.
    x = np.array([[1.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    f = objectives(inst, Solution(y=np.array([True, True]), x=x))
    assert f.tolist() == [140.0, 1.0 + 2.0 + 30.0]   # C_ij is not multiplied by demand


def test_split_customer_is_charged_by_share():
    inst = tiny_instance()
    x = np.array([[1.0, 1.0, 0.4],      # facility 0 serves 2 of customer 2's 5 units
                  [0.0, 0.0, 0.6]])
    f = objectives(inst, Solution(y=np.array([True, True]), x=x))
    assert np.isclose(f[1], 1.0 + 2.0 + 0.4 * 3.0 + 0.6 * 30.0)


def test_evaluator_counts_and_returns_repaired_chromosome():
    inst = load_instance("cap101")
    ev = Evaluator(inst)
    y, f = ev(np.zeros(inst.m, dtype=bool))
    y2, _ = ev(y)
    assert ev.n_evals == 2
    assert y.any() and np.array_equal(y, y2)
    assert f.shape == (2,) and (f >= 0).all()


def test_sum_is_never_below_known_optimum():
    # f1 + f2 of any feasible solution cannot beat Beasley's single-objective optimum.
    opt = load_optimal_values()
    rng = np.random.default_rng(0)
    for name in ["cap41", "cap101", "cap121"]:
        inst = load_instance(name)
        for _ in range(30):
            f = objectives(inst, decode(inst, rng.random(inst.m) < 0.5))
            assert f.sum() >= opt[name] - 1e-6
