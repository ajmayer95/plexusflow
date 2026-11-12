"""Graph region classes in the carved reconstruction network (plexusflow.reconcile.regions, network.carved_domain):
untagged graphs are unchanged by the topology option, a topology-uncertain region is a mass-conserving black box, a region
touching an open region is accounted exactly as if it were tagged open, and the tags are mutually exclusive."""
import networkx as nx
import numpy as np

from plexusflow.reconcile import regions as RG
from plexusflow.reconcile.network import carved_domain, emergent, project


def grid_graph(nx_=10, ny=8):
    """Grid with a DA anchor (boundary_type 'source') wired to the left column and two vein anchors on the right."""
    G = nx.Graph()
    idx = lambda i, j: i * ny + j
    for i in range(nx_):
        for j in range(ny):
            G.add_node(idx(i, j), x=float(10 * i), y=float(10 * j))
    for i in range(nx_):
        for j in range(ny):
            if i + 1 < nx_:
                G.add_edge(idx(i, j), idx(i + 1, j))
            if j + 1 < ny:
                G.add_edge(idx(i, j), idx(i, j + 1))
    A = nx_ * ny; G.add_node(A, x=-10.0, y=35.0, boundary_type="source")
    for j in range(ny):
        G.add_edge(A, idx(0, j))
    G.nodes[idx(nx_ - 1, 1)]["boundary_type"] = "sink"; G.nodes[idx(nx_ - 1, ny - 2)]["boundary_type"] = "sink"
    for u, v in G.edges:
        G.edges[u, v].update(radius=10.0, length=10.0)
    return G, idx


def tag(G, edges, cls):
    for e in edges:
        RG.set_region(G.edges[e], cls)


def data(dom, seed=0):
    return np.random.default_rng(seed).normal(0, 1, len(dom["edges"]))


def test_untagged_graph_is_unchanged_by_the_topology_option():
    G, _ = grid_graph()
    a, b = carved_domain(G, topology=True), carved_domain(G, topology=False)
    assert a["edges"] == b["edges"] and np.array_equal(a["bnd"], b["bnd"]) and a["B"].shape == b["B"].shape
    assert a["regions"] == [] and len(a["topo_cut"]) == 0


def test_interior_topology_region_is_a_conserving_black_box():
    G, idx = grid_graph()
    block = [(idx(i, j), idx(i + 1, j)) for i in (4, 5) for j in (3, 4, 5)] + [(idx(i, j), idx(i, j + 1)) for i in (4, 5, 6) for j in (3, 4)]
    tag(G, block, RG.TOPOLOGY)
    dom = carved_domain(G)
    assert len(dom["regions"]) == 1
    reg = dom["regions"][0]
    assert not reg["leaky"] and reg["row"] == dom["n_real"] and dom["B"].shape[0] == dom["n_real"] + 1
    assert not set(map(frozenset, block)) & set(map(frozenset, dom["edges"]))          # region edges removed
    ix = {n: k for k, n in enumerate(dom["nodes"])}
    cut = [ix[n] for n in reg["cut"]]
    assert set(cut) <= set(dom["topo_cut"]) <= set(dom["bnd"])                         # cut nodes exchange freely
    Q = project(data(dom), np.ones(len(dom["edges"])), dom)
    f = dom["B"][:dom["n_real"]] @ Q
    assert abs(f[cut].sum()) < 1e-9                                                     # …but the region's net exchange is zero
    assert np.max(np.abs(f[cut])) > 1e-3                                                # and flow does pass through it
    interior = np.setdiff1d(np.arange(dom["n_real"]), dom["bnd"])
    assert np.max(np.abs(f[interior])) < 1e-9


def test_topology_region_touching_an_open_region_counts_as_open():
    def build(cls):
        G, idx = grid_graph()
        tag(G, [(idx(0, j), idx(1, j)) for j in (2, 3, 4)], RG.OPEN)                   # open region beside the DA
        tag(G, [(idx(1, j), idx(2, j)) for j in (2, 3, 4)] + [(idx(1, 2), idx(1, 3)), (idx(1, 3), idx(1, 4))], cls)
        return carved_domain(G)
    t, o = build(RG.TOPOLOGY), build(RG.OPEN)
    assert t["regions"][0]["leaky"] and t["B"].shape[0] == t["n_real"]                 # no conservation row
    assert t["edges"] == o["edges"]
    inj = lambda d: {d["nodes"][i] for i in d["inj"]}
    assert inj(t) == inj(o)
    y = data(t); w = np.ones(len(y))
    Qt, Qo = project(y, w, t), project(y, w, o)
    assert np.allclose(Qt, Qo, atol=1e-10)
    assert np.isclose(emergent(Qt, t)[t["inj"]].sum(), emergent(Qo, o)[o["inj"]].sum())


def test_region_tags_are_exclusive_and_write_the_legacy_key():
    d = {}
    RG.set_region(d, RG.OPEN)
    assert RG.is_open(d) and d[RG.LEGACY_OPEN_KEY] and not RG.is_topology(d)
    RG.set_region(d, RG.TOPOLOGY)
    assert RG.is_topology(d) and not RG.is_open(d) and RG.LEGACY_OPEN_KEY not in d
    RG.set_region(d, None)
    assert RG.region_class(d) is None
    assert RG.region_class({"radial_uncertain": True}) == RG.OPEN                       # legacy graphs still read as open
