import numpy as np
import pytest

from cflp_moea.data_loader import load_instance
from cflp_moea.representation import (check_feasible, decode, facility_load,
                                      oversized_customers, random_chromosome)

INSTANCES = ["cap41", "cap42", "cap101", "cap102", "cap121", "cap122"]


@pytest.mark.parametrize("name", INSTANCES)
def test_random_chromosomes_decode_feasible(name):
    inst = load_instance(name)
    rng = np.random.default_rng(4610)
    for p_open in (0.0, 0.1, 0.5, 1.0):
        for _ in range(50):
            sol = decode(inst, random_chromosome(inst, rng, p_open))
            assert check_feasible(inst, sol) == []


@pytest.mark.parametrize("name", INSTANCES)
def test_no_open_facility_is_unused(name):
    inst = load_instance(name)
    sol = decode(inst, np.ones(inst.m, dtype=bool))
    assert (sol.x[sol.y].sum(axis=1) > 0).all()


def test_only_cap41_cap42_have_oversized_customers():
    counts = {n: int(oversized_customers(load_instance(n)).sum()) for n in INSTANCES}
    assert counts == {"cap41": 2, "cap42": 2, "cap101": 0, "cap102": 0, "cap121": 0, "cap122": 0}


def test_cap41_oversized_customers_are_split_within_capacity():
    inst = load_instance("cap41")
    sol = decode(inst, np.zeros(inst.m, dtype=bool))
    for j in np.flatnonzero(oversized_customers(inst)):
        served = sol.x[:, j] * inst.demand[j]
        assert (served > 0).sum() >= 2
        assert np.isclose(served.sum(), inst.demand[j])
    assert (facility_load(inst, sol) <= inst.capacity + 1e-6).all()


def test_decode_is_deterministic_and_does_not_modify_input():
    inst = load_instance("cap101")
    chrom = random_chromosome(inst, np.random.default_rng(1))
    before = chrom.copy()
    a, b = decode(inst, chrom), decode(inst, chrom)
    assert np.array_equal(chrom, before)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.x, b.x)


def test_single_open_facility_serves_everyone_when_it_fits():
    # in cap101 one facility can hold all demand
    inst = load_instance("cap101")
    chrom = np.zeros(inst.m, dtype=bool)
    chrom[3] = True
    sol = decode(inst, chrom)
    assert sol.y.sum() == 1 and sol.y[3]
    assert (sol.x[3] == 1).all()
