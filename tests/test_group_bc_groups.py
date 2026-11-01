"""Boundary grouping of the (D, η) estimate: one group index per boundary node, any number of V nodes."""
import numpy as np

from plexusmech.ac.group_bc import groups


def _ctx(side, side_out):
    n_in, n_out = len(side), len(side_out)
    return {"dom": {"side": np.array(side), "side_out": np.array(side_out), "out": np.arange(n_in, n_in + n_out),
                    "bnd": np.arange(n_in + n_out)}}


def test_two_veins_match_old_grouping():
    g = groups(_ctx(["L", "R", "L"], ["L", "R"]))
    assert g.tolist() == [0, 1, 0, 2, 3]


def test_many_veins_one_index_per_boundary_node():
    side_out = ["L"] * 17 + ["R"] * 22
    g = groups(_ctx(["L"] * 9 + ["R"] * 6, side_out))
    assert len(g) == 15 + 39
    assert (g[15:] == np.where(np.array(side_out) == "L", 2, 3)).all()


def test_missing_side_out_with_many_veins_raises():
    C = _ctx(["L", "R"], ["L", "R", "R"]); del C["dom"]["side_out"]
    try:
        groups(C)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
