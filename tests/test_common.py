import numpy as np

from cflp_moea.algorithms.common import (binary_tournament, dominates, non_dominated_mask,
                                         unique_non_dominated)


def test_dominates():
    assert dominates(np.array([1, 2]), np.array([2, 2]))
    assert not dominates(np.array([1, 2]), np.array([1, 2]))
    assert not dominates(np.array([1, 3]), np.array([2, 2]))


def test_non_dominated_mask():
    F = np.array([[1, 5], [2, 2], [5, 1], [3, 3], [2, 2]])
    assert non_dominated_mask(F).tolist() == [True, True, True, False, True]


def test_unique_non_dominated_removes_duplicates():
    F = np.array([[1, 5], [2, 2], [3, 3], [2, 2]])
    X = np.arange(4).reshape(4, 1)
    Xn, Fn = unique_non_dominated(X, F)
    assert Fn.tolist() == [[1, 5], [2, 2]] and Xn.ravel().tolist() == [0, 1]


def test_binary_tournament_prefers_lower_score():
    rng = np.random.default_rng(0)
    score = np.array([0.0, 1.0, 2.0, 3.0])
    picks = binary_tournament(score, 10_000, rng)
    counts = np.bincount(picks, minlength=4)
    assert counts[0] > counts[1] > counts[2] > counts[3]
