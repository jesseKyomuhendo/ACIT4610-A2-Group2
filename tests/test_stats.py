import numpy as np

from cflp_moea.stats import a12, compare, summary


def test_summary():
    s = summary([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and s["best"] == 3.0 and s["worst"] == 1.0
    assert np.isclose(s["std"], 1.0)


def test_a12():
    assert a12([2, 3], [0, 1]) == 1.0
    assert a12([1, 1], [1, 1]) == 0.5


def test_compare_finds_clear_difference():
    a = np.linspace(0.90, 0.91, 10)
    b = np.linspace(0.80, 0.81, 10)
    assert compare(a, b)["winner"] == "first"
    assert compare(b, a)["winner"] == "second"


def test_compare_identical_results():
    res = compare([0.76] * 10, [0.76] * 10)
    assert res["p_value"] == 1.0 and res["winner"] == "no significant difference"
