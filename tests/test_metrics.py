import numpy as np

from cflp_moea.metrics import REF_POINT, bounds, hypervolume_2d, n_non_dominated, normalise


def test_hypervolume_hand_example():
    # rectangles: (2-1)*(4-2) + (3-2)*(4-1) = 2 + 3
    F = np.array([[2.0, 1.0], [1.0, 2.0]])
    assert np.isclose(hypervolume_2d(F, np.array([3.0, 4.0])), 5.0)


def test_hypervolume_ignores_dominated_duplicate_and_outside_points():
    ref = np.array([3.0, 4.0])
    F = np.array([[1.0, 2.0], [2.0, 1.0], [2.0, 3.0], [1.0, 2.0], [5.0, 0.5]])
    assert np.isclose(hypervolume_2d(F, ref), 5.0)


def test_hypervolume_single_point_and_empty():
    assert np.isclose(hypervolume_2d(np.array([[0.0, 0.0]])), 1.1 * 1.1)
    assert hypervolume_2d(np.array([[2.0, 2.0]])) == 0.0


def test_normalise_with_shared_bounds():
    a = np.array([[100.0, 900.0], [300.0, 500.0]])
    b = np.array([[200.0, 700.0]])
    lo, hi = bounds([a, b])
    assert lo.tolist() == [100, 500] and hi.tolist() == [300, 900]
    na, nb = normalise(a, lo, hi), normalise(b, lo, hi)
    assert na.tolist() == [[0, 1], [1, 0]] and nb.tolist() == [[0.5, 0.5]]
    assert (np.vstack([na, nb]) < REF_POINT).all()


def test_better_front_has_larger_hypervolume():
    worse = np.array([[0.2, 0.8], [0.8, 0.2]])
    better = worse - 0.1
    assert hypervolume_2d(better) > hypervolume_2d(worse)


def test_n_non_dominated():
    F = np.array([[1, 5], [2, 2], [2, 2], [3, 3], [5, 1]])
    assert n_non_dominated(F) == 3
