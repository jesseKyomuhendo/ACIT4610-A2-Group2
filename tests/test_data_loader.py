import numpy as np
import pytest

from cflp_moea.data_loader import load_instance

EXPECTED = {"cap41": 16, "cap42": 16, "cap101": 25, "cap102": 25, "cap121": 50, "cap122": 50}


@pytest.mark.parametrize("name,m", EXPECTED.items())
def test_dimensions(name, m):
    inst = load_instance(name)
    assert (inst.m, inst.n) == (m, 50)
    assert inst.alloc_cost.shape == (m, 50)
    assert inst.capacity.sum() >= inst.demand.sum()


def test_cap41_spot_values():
    inst = load_instance("cap41")
    assert inst.capacity[0] == 5000 and inst.fixed_cost[0] == 7500
    assert inst.fixed_cost[10] == 0            # facility 11 really has fixed cost 0 in the file
    assert inst.demand[0] == 146
    assert np.isclose(inst.alloc_cost[0, 0], 6739.725)
    assert np.isclose(inst.alloc_cost[15, 0], 6051.7)
    assert inst.demand[-1] == 222
