"""A release graph keeps exactly plexusflow.release.RELEASE_FIELDS."""
import networkx as nx
import numpy as np

from plexusflow.release import RELEASE_FIELDS, to_release_schema


def _graph():
    G = nx.Graph()
    G.graph.update(tile_f0_piv={1: 2.7}, tile_gauges={}, sim_meta={})
    G.add_node(1, x=0.0, y=0.0, boundary_type="source", bc_harmonics=[1, 2])
    G.add_node(2, x=1.0, y=0.0)
    m = dict(tile_id=1, flow_from=1, flow_to=2, Q_t=np.zeros(8), Q_profile=0.1, R_profile_um=10.0, Q_direct=0.2, v_t=np.zeros(8),
             harm=dict(f0_hz=2.7, mean_Q=0.1, harmonics=[], a0=0.1), quality=dict(profile_ok=True, f=0.5, align=0.9, p=0.3, n_kept=4),
             snr=dict(dc_db=10.0, h1_db=5.0), motion=dict(z_dc=1.0, p_ac=0.1, eta_osc=0.2), geom=dict(R_um=9.0), qc=dict(passed_qc=True))
    G.add_edge(1, 2, path=np.zeros((2, 2)), radius=3.0, length=5.0, radial_uncertain=False, _h_amp_H1=1.0, seg_chain=[1],
               measurements_piv=[m])
    return G


def test_only_release_fields_remain():
    G = to_release_schema(_graph())
    assert set(G.graph) <= RELEASE_FIELDS["graph"]
    for _, d in G.nodes(data=True):
        assert set(d) <= RELEASE_FIELDS["node"]
    for *_, d in G.edges(data=True):
        assert set(d) <= RELEASE_FIELDS["edge"]
        for m in d["measurements_piv"]:
            assert set(m) <= RELEASE_FIELDS["measurement"]
            for sub in ("harm", "quality", "snr", "motion"):
                assert set(m[sub]) <= RELEASE_FIELDS[f"measurement.{sub}"]


def test_kept_values_unchanged():
    G = to_release_schema(_graph())
    m = G.edges[1, 2]["measurements_piv"][0]
    assert m["Q_profile"] == 0.1 and m["quality"]["profile_ok"] is True and m["harm"]["f0_hz"] == 2.7
    assert G.nodes[1]["boundary_type"] == "source" and G.graph["tile_f0_piv"] == {1: 2.7}
