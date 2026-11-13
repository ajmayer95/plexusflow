"""Gross-error detection (plexusflow.reconcile.qc) on a small grid network with known flow: a corrupted measurement, a bad
repeat recording, a junction imbalance, a circulation and an over-thick vessel are each flagged; clean data are not."""
import networkx as nx
import numpy as np

from plexusflow.reconcile.network import carved_domain, passive, project
from plexusflow.reconcile.qc import measurement_test, repeat_test, nodal_test, loop_flow, radius_test, run_qc

C0 = 0.2
MODEL = dict(c0=C0, c_p=0.0, c_a=0.0, c_T={}, profile_term="p", sigma0=1e-4)


def grid(nx_=9, ny=7):
    G = nx.Graph(); idx = lambda i, j: i * ny + j
    for i in range(nx_):
        for j in range(ny):
            G.add_node(idx(i, j), x=float(20 * i), y=float(20 * j))
    for i in range(nx_):
        for j in range(ny):
            for a, b in (((i, j), (i + 1, j)), ((i, j), (i, j + 1))):
                if b[0] < nx_ and b[1] < ny:
                    u, v = idx(*a), idx(*b)
                    G.add_edge(u, v, radius=10.0, length=20.0,
                               path=np.array([[G.nodes[u]["y"], G.nodes[u]["x"]], [G.nodes[v]["y"], G.nodes[v]["x"]]], float))
    A = nx_ * ny; G.add_node(A, x=-20.0, y=60.0, boundary_type="source")
    for j in range(ny):
        G.add_edge(A, idx(0, j), radius=10.0, length=20.0, path=np.zeros((2, 2)))
    G.nodes[idx(nx_ - 1, 1)]["boundary_type"] = "sink"; G.nodes[idx(nx_ - 1, ny - 2)]["boundary_type"] = "sink"
    return G, idx


def field(G, dom, noise=0.0, seed=0):
    """True passive flow, measured with multiplicative noise; reconstruction = weighted projection with σ = c|y|."""
    K = np.ones(len(dom["edges"])); fb = np.zeros(dom["B"].shape[0]); fb[dom["inj"]] = 1.0; fb[dom["out"]] = -len(dom["inj"]) / 2
    Q = passive(K, fb, dom); rng = np.random.default_rng(seed)
    y = Q * np.exp(noise * rng.standard_normal(len(Q)))
    return Q, y


def dc_dict(dom, y, tile=1):
    s0 = 1e-4; sig = np.sqrt((C0 * y) ** 2 + s0 ** 2); Qp = project(y, 1 / sig ** 2, dom)
    return dict(edges=dom["edges"], Qm=y, Qp=Qp, sigma=np.sqrt((C0 * Qp) ** 2 + s0 ** 2), imaged=np.ones(len(y), bool), c=np.full(len(y), C0),
                sigma0_floor=s0, s_between=np.zeros(len(y)), tile=np.full(len(y), tile))


def test_measurement_test_flags_a_corrupted_vessel_only():
    G, _ = grid(); dom = carved_domain(G); Q, y = field(G, dom, noise=0.05)
    big = int(np.argmax(np.abs(Q))); y = y.copy(); y[big] *= 8.0                     # a gross over-read on a well-constrained vessel
    z, rev, r = measurement_test(dc_dict(dom, y), dom)
    assert abs(z[big]) > 4
    others = np.delete(np.abs(np.nan_to_num(z)), big)
    assert np.percentile(others, 95) < 3 and not rev.any()


def test_repeat_test_flags_the_odd_recording_out():
    e = (0, 1); M = {(t, e): dict(Q=q, tile=t, p=1.0, f=1.0, align=1.0) for t, q in ((1, 1.0), (2, 1.05), (3, 0.97), (4, 0.10))}
    out = repeat_test(M, MODEL, 1e-4)
    assert abs(out[(4, e)][0]) > 4 and all(abs(out[(t, e)][0]) < 2 for t in (1, 2, 3))
    M2 = {(1, e): dict(Q=1.0, tile=1, p=1.0, f=1.0, align=1.0), (2, e): dict(Q=-1.0, tile=2, p=1.0, f=1.0, align=1.0)}
    assert repeat_test(M2, MODEL, 1e-4)[(2, e)][1]                                    # resolved opposite directions


def test_nodal_test_flags_an_imbalanced_junction():
    G, idx = grid(); dom = carved_domain(G); Q, y = field(G, dom)
    n = idx(4, 3); i = dom["nodes"].index(n); cols = dom["B"][i].indices
    k = cols[np.argmax(np.abs(y[cols]))]; y = y.copy(); y[k] = -y[k]                 # largest incident vessel read reversed
    z = nodal_test(dc_dict(dom, y), dom)
    assert abs(z[n]) > 4 and np.percentile([abs(v) for m, v in z.items() if m != n and abs(v) > 0], 90) < 4


def test_loop_flow_finds_an_added_circulation_and_nothing_in_potential_flow():
    G, idx = grid(); dom = carved_domain(G); Q, _ = field(G, dom)
    assert loop_flow(dict(edges=dom["edges"], Qp=Q)).max() < 1e-9                    # pressure-driven flow has no directed loops
    ring = [(idx(3, 2), idx(4, 2)), (idx(4, 2), idx(4, 3)), (idx(4, 3), idx(3, 3)), (idx(3, 3), idx(3, 2))]
    ix = {tuple(e): k for k, e in enumerate(dom["edges"])}; Qc = Q.copy()
    for u, v in ring:                                                               # +5 nL/s around the ring, in each edge's frame
        if (u, v) in ix:
            Qc[ix[(u, v)]] += 5.0
        else:
            Qc[ix[(v, u)]] -= 5.0
    pe = loop_flow(dict(edges=dom["edges"], Qp=Qc))
    on = [ix[(u, v)] if (u, v) in ix else ix[(v, u)] for u, v in ring]
    assert pe[on].min() > 4.0 and np.delete(pe, on).max() < 1.0


def test_radius_test_flags_a_vessel_thicker_than_all_neighbours():
    G, idx = grid(); e = (idx(4, 3), idx(5, 3)); G.edges[e]["radius"] = 30.0
    dom = carved_domain(G); _, _, thick = radius_test(G, dict(edges=dom["edges"]))
    k = [tuple(x) for x in dom["edges"]].index(e)
    assert thick[k] and thick.sum() == 1


def test_run_qc_groups_flags_into_ranked_hotspots():
    G, _ = grid(); dom = carved_domain(G); Q, y = field(G, dom, noise=0.05)
    big = int(np.argmax(np.abs(Q))); y = y.copy(); y[big] *= 8.0
    M = {(1, tuple(e)): dict(Q=float(q), tile=1, p=1.0, f=1.0, align=1.0) for e, q in zip(dom["edges"], y)}
    rows, hot, summ = run_qc(G, dc_dict(dom, y), dom, M, MODEL)
    assert any(r["kind"] == "measurement" and tuple(r["edge"]) == tuple(dom["edges"][big]) for r in rows)
    assert hot and hot[0]["hotspot"] == 1 and all(r["hotspot"] >= 1 for r in rows)
