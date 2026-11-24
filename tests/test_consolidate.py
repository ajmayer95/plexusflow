"""Graph consolidation (plexusflow.edit.consolidate): a speck loop in one vessel becomes one smooth vessel on the mask, long
holes between real channels are kept, and redraws that double back are rejected."""
import networkx as nx
import numpy as np

from plexusflow.edit.consolidate import consolidate, path_problem, pillar_holes

H_IMG = 120


def band_mask(hole=None):
    m = np.zeros((H_IMG, 200), bool); m[50:71, 5:195] = True                       # one horizontal vessel, radius ~10 px
    if hole is not None:
        r0, r1, c0, c1 = hole; m[r0:r1, c0:c1] = False
    return m


def xy(col, row):
    return dict(x=float(col), y=float(H_IMG - row))


def speck_loop_graph():
    """A ── L ─ T ─ R ── B, with L ─ Bt ─ R: the loop net3 draws around a speck at (row 60, col 100)."""
    G = nx.Graph(); pos = dict(A=(15, 60), L=(92, 60), T=(100, 54), R=(108, 60), Bt=(100, 66), B=(185, 60))
    ids = {k: i for i, k in enumerate(pos)}
    for k, (c, r) in pos.items():
        G.add_node(ids[k], **xy(c, r))
    for a, b in [("A", "L"), ("L", "T"), ("T", "R"), ("R", "Bt"), ("Bt", "L"), ("R", "B")]:
        pa, pb = np.array([pos[a][0], H_IMG - pos[a][1]], float), np.array([pos[b][0], H_IMG - pos[b][1]], float)
        G.add_edge(ids[a], ids[b], path=np.linspace(pa, pb, 20), radius=10.0, length=float(np.linalg.norm(pb - pa)))
    G.edges[ids["A"], ids["L"]]["measurements_piv"] = [dict(tile_id=1, flow_from=ids["A"], flow_to=ids["L"], quality=dict(profile_ok=True, p=0.9))]
    return G, ids


def test_speck_loop_becomes_one_vessel():
    m = band_mask(hole=(59, 62, 99, 102)); G, ids = speck_loop_graph(); n0 = G.number_of_edges()
    H, emap, rep, mf = consolidate(G, m)
    assert G.number_of_edges() == n0                                                    # input untouched
    assert rep["holes_filled"] == 1 and rep["one_vessel"] == 1 and not rep["review"]
    A, B = ids["A"], ids["B"]
    assert H.has_edge(A, B) and set(H.nodes) == {A, B} and H.edges[A, B]["consolidated"]
    p = np.asarray(H.edges[A, B]["path"])
    rows, cols = np.round(H_IMG - p[:, 1]).astype(int), np.round(p[:, 0]).astype(int)
    assert mf[rows, cols].all() and path_problem(p) is None                             # on the mask, no kinks
    assert 8 <= H.edges[A, B]["radius"] <= 12
    rec = H.edges[A, B]["measurements_piv"][0]
    assert rec["flow_from"] == A and rec["flow_to"] == B                                # recording keeps its direction
    assert set(emap[(A, ids["L"])]) == {A, B}


def test_long_hole_between_channels_is_kept_and_specks_are_filled():
    m = band_mask(hole=(58, 63, 40, 160))                                              # 120 px long, side channels ~8 px
    m[52:54, 20:23] = False                                                             # plus a 6 px speck
    lab, keep, lt = pillar_holes(m)
    long_ = lab[60, 100]; speck = lab[52, 21]
    assert not keep[long_] and lt[long_] > 1 and keep[speck]


def test_path_problem_flags_hairpins_and_self_loops():
    t = np.linspace(0, 1, 60)
    assert path_problem(np.column_stack([t * 60, np.zeros_like(t)])) is None
    hair = np.vstack([np.column_stack([np.linspace(0, 30, 31), np.zeros(31)]), np.column_stack([np.linspace(29, 10, 20), np.full(20, 1.0)])])
    assert path_problem(hair) == "hairpin"
    assert path_problem(np.column_stack([t * 60, np.zeros_like(t)]), on_mask_frac=0.5) == "off-mask"
