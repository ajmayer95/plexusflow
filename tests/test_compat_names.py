"""Legacy flux-estimator names are upgraded on load; current names pass through unchanged."""
import pickle
import networkx as nx
from plexusflow.compat import loads_graph, upgrade_graph, sidecar_dir, sidecar_inputs, MEASUREMENT_KEYS


def _legacy_graph():
    G = nx.Graph()
    m = {old: float(i + 1) for i, old in enumerate(MEASUREMENT_KEYS)}
    m.update(quality={"kl_ok": True, "f": 0.5}, prov={"kl_mode": "sign"}, Q_nls={"best": 1.0, "best_key": "kl"})
    G.add_edge(1, 2, measurements_piv=[m])
    G.graph["measurement_batch"] = {"kl_sidecar": "x"}
    return G


def test_legacy_graph_is_upgraded_on_load():
    G = loads_graph(pickle.dumps(_legacy_graph()))
    m = G.edges[1, 2]["measurements_piv"][0]
    for i, new in enumerate(MEASUREMENT_KEYS.values()):
        assert m[new] == float(i + 1)
    assert not set(MEASUREMENT_KEYS) & set(m)
    assert m["quality"] == {"profile_ok": True, "f": 0.5} and m["prov"] == {"profile_mode": "sign"}
    assert m["Q_nls"]["best_key"] == "profile" and G.graph["measurement_batch"] == {"profile_sidecar": "x"}
    assert upgrade_graph(G) == 0                       # idempotent


def test_sidecars_under_either_name(tmp_path):
    assert sidecar_dir(tmp_path).name == "batch_profile_inputs"
    (tmp_path / "batch_kl_inputs").mkdir()
    assert sidecar_dir(tmp_path).name == "batch_kl_inputs"
    (tmp_path / "batch_profile_inputs").mkdir()
    assert sidecar_dir(tmp_path).name == "batch_profile_inputs"
    assert sidecar_inputs({"kl_inputs": {1: 2}}) == {1: 2} and sidecar_inputs({"profile_inputs": {3: 4}}) == {3: 4}
