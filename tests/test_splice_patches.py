"""Region re-vectorisation (plexusflow.edit.splice) and the stage patch log (plexusflow.edit.patches).

A synthetic plexus: a 7 × 7 grid of 9-px-wide vessels on a 400 × 400 mask.  A stand-in vectoriser (net3 is not needed)
turns every grid segment that is fully on the crop mask into a chain of skeleton nodes, in crop coordinates with y up —
the same convention as net3's output."""
import gzip
import pickle

import numpy as np
import networkx as nx
import pytest

from plexusflow.analyze.condense import condense_graph
from plexusflow.edit import patches as PT
from plexusflow.edit.splice import splice_region, prep_mask
from plexusflow.reconcile import regions as RG

H = W = 400
LINES = list(range(50, 351, 50))


def grid_mask():
    m = np.zeros((H, W), bool)
    for c in LINES:
        m[50 - 4:350 + 5, c - 4:c + 5] = True        # vertical vessel at x = c
        m[H - c - 4:H - c + 5, 50 - 4:350 + 5] = True  # horizontal vessel at y_up = c
    return m


def fake_vectorizer(crop):
    """Skeleton chains along the grid lines that lie on the crop (2-px node spacing), crop coordinates (x, y_up)."""
    h, w = crop.shape; R = nx.Graph(); ids = {}
    def node(x, y):
        k = (round(x, 3), round(y, 3))
        if k not in ids:
            ids[k] = len(ids); R.add_node(ids[k], x=float(x), y=float(y), radius=4.5)
        return ids[k]
    on = lambda x, y: 0 <= int(round(h - y)) < h and 0 <= int(round(x)) < w and crop[int(round(h - y)), int(round(x))]
    # crop offset is unknown to the vectoriser: work on the crop's own pixels; grid lines are where the mask is a ridge
    rows = [r for r in range(h) if crop[r].sum() > 0.6 * w and (r == 0 or crop[r].sum() >= crop[r - 1].sum()) and (r == h - 1 or crop[r].sum() >= crop[r + 1].sum())]
    cols = [c for c in range(w) if crop[:, c].sum() > 0.6 * h and (c == 0 or crop[:, c].sum() >= crop[:, c - 1].sum()) and (c == w - 1 or crop[:, c].sum() >= crop[:, c + 1].sum())]
    def centres(idx):
        out, run = [], []
        for i in idx:
            if run and i != run[-1] + 1:
                out.append(np.mean(run)); run = []
            run.append(i)
        if run:
            out.append(np.mean(run))
        return out
    for r in centres(rows):
        y = h - r; xs = np.arange(0, w, 2.0)
        prev = None
        for x in xs:
            if on(x, y):
                n = node(x, y)
                if prev is not None: R.add_edge(prev, n)
                prev = n
            else:
                prev = None
    for c in centres(cols):
        ys = np.arange(0, h, 2.0); prev = None
        for y in ys:
            if on(c, y):
                n = node(c, y)
                if prev is not None: R.add_edge(prev, n)
                prev = n
            else:
                prev = None
    # join horizontal and vertical chains at crossings: snap nodes within 1.5 px
    pts = {n: (R.nodes[n]["x"], R.nodes[n]["y"]) for n in R.nodes}
    from scipy.spatial import cKDTree
    keys = list(pts); T = cKDTree(np.array([pts[k] for k in keys]))
    for a, b in T.query_pairs(1.5):
        if not R.has_edge(keys[a], keys[b]):
            R.add_edge(keys[a], keys[b])
    return R


def build_graph(m):
    R = fake_vectorizer(m)
    V, _ = condense_graph(R)
    for u, v, d in V.edges(data=True):
        d.update(radius=4.5, length=float(np.hypot(*np.diff(np.asarray(d["path"]), axis=0).T).sum()),
                 measurements_piv=[dict(tile_id=1, Q=0.1)], harmonic_class="x")
    return V


@pytest.fixture
def setup():
    m = grid_mask(); return m, build_graph(m)


BOX = np.array([[140, 140], [260, 140], [260, 260], [140, 260]], float)


def test_noop_splice_reproduces_the_region(setup):
    m, G = setup
    r = splice_region(G, m, BOX, vectorizer=fake_vectorizer, margin=60)
    assert r.report["dead_ends"] == [] and r.report["orphaned_old_nodes"] == []
    rem = set(r.removed_edges)
    for e in G.edges:                                                      # outside the region: untouched (same objects)
        if e not in rem:
            assert r.G.has_edge(*e) and r.G.edges[e] is not None and r.G.edges[e]["measurements_piv"] == G.edges[e]["measurements_piv"]
    L_old = sum(G.edges[e]["length"] for e in r.removed_edges); L_new = sum(r.G.edges[e]["length"] for e in r.new_edges)
    assert abs(L_new - L_old) / L_old < 0.03
    assert all(r.G.edges[e]["measurements_piv"] == [] and r.G.edges[e]["spliced"] for e in r.new_edges)
    assert nx.number_connected_components(r.G) == nx.number_connected_components(G)
    assert all(abs(r.G.edges[e]["radius"] - 4.5) <= 1.0 for e in r.new_edges)   # median distance to wall of a 9-px vessel


def test_mask_edit_removes_a_vessel_inside_the_region(setup):
    m, G = setup
    m2 = m.copy(); m2[H - 200 - 6:H - 200 + 7, 160:241] = False          # cut the horizontal vessel y = 200 between x 160–240
    m2[:, 200 - 4:200 + 5] |= m[:, 200 - 4:200 + 5]                        # …but keep the vertical one at x = 200 intact
    r = splice_region(G, m2, BOX, vectorizer=fake_vectorizer, margin=60)
    on_line = [e for e in r.new_edges if np.all(np.abs(np.asarray(r.G.edges[e]["path"])[:, 1] - 200) < 2)
               and np.ptp(np.asarray(r.G.edges[e]["path"])[:, 0]) > 30]
    assert not any(np.any((np.asarray(r.G.edges[e]["path"])[:, 0] > 170) & (np.asarray(r.G.edges[e]["path"])[:, 0] < 190)) for e in on_line)
    assert len(r.new_edges) < len(r.removed_edges) + 4


def test_region_with_an_anchor_is_refused(setup):
    m, G = setup
    n = min(G.nodes, key=lambda k: (G.nodes[k]["x"] - 200) ** 2 + (G.nodes[k]["y"] - 200) ** 2)
    G.nodes[n]["boundary_type"] = "sink"
    with pytest.raises(ValueError, match="anchor"):
        splice_region(G, m, BOX, vectorizer=fake_vectorizer)


def test_new_vessels_inherit_region_tags(setup):
    m, G = setup
    for e in G.edges:
        RG.set_region(G.edges[e], RG.TOPOLOGY)
    r = splice_region(G, m, BOX, vectorizer=fake_vectorizer, margin=60)
    assert all(RG.is_topology(r.G.edges[e]) for e in r.new_edges)


def test_prep_mask_fills_small_holes_only():
    b = np.ones((50, 50), bool); b[10:12, 10:12] = False; b[30:40, 30:40] = False
    p = prep_mask(b, fill_holes=20)
    assert p[10:12, 10:12].all() and not p[31:39, 31:39].any()          # (slit closing trims the big hole's corners, as the stage prep does)


def test_diff_apply_revert_round_trip(setup):
    m, G = setup
    r = splice_region(G, m, BOX, vectorizer=fake_vectorizer, margin=60)
    d = PT.diff(G, r.G)
    G2 = G.copy(); PT.apply(G2, d)
    assert PT.is_empty(PT.diff(G2, r.G))
    PT.revert(G2, d)
    assert PT.is_empty(PT.diff(G2, G))


def test_patch_log_accept_revert_and_conflicts(tmp_path, setup):
    import tifffile
    m, G = setup
    gp = tmp_path / "graph" / "g.gpickle"; gp.parent.mkdir(); gp.write_bytes(pickle.dumps(G))
    mp = tmp_path / "mask" / "mask.tif"; mp.parent.mkdir(); tifffile.imwrite(mp, (m * 255).astype(np.uint8))
    log = PT.PatchLog(gp, mp)
    m2 = m.copy(); m2[H - 200 - 6:H - 200 + 7, 160:241] = False; m2[:, 196:205] |= m[:, 196:205]
    r1 = splice_region(G, m2, BOX, vectorizer=fake_vectorizer, margin=60)
    e1 = log.accept(G, r1.G, kind="revectorize", note="test", polygon=BOX, mask_before=m, mask_after=m2, report=r1.report)
    assert e1["id"] == "0001" and e1["mask"]["n_removed"] > 0
    Gd = pickle.loads(gp.read_bytes()); assert PT.is_empty(PT.diff(Gd, r1.G))
    assert (tifffile.imread(mp) > 127).sum() == m2.sum()
    assert len(list((tmp_path / "graph" / "patches").glob("base_*.gpickle"))) == 1
    # a second, overlapping patch blocks reverting the first
    box2 = BOX + 30
    r2 = splice_region(Gd, m2, box2, vectorizer=fake_vectorizer, margin=60)
    log.accept(Gd, r2.G, kind="revectorize", polygon=box2)
    assert log.conflicts("0001") == ["0002"]
    with pytest.raises(ValueError, match="later patch"):
        log.revert(pickle.loads(gp.read_bytes()), "0001")
    G3 = log.revert(pickle.loads(gp.read_bytes()), "0002")
    G4 = log.revert(G3, "0001")
    assert PT.is_empty(PT.diff(G4, G)) and PT.is_empty(PT.diff(pickle.loads(gp.read_bytes()), G))
    assert (tifffile.imread(mp) > 127).sum() == m.sum()
    assert [e["status"] for e in log.entries()] == ["reverted", "reverted"]


def test_diff_ignores_nan_in_nested_measurements():
    G = nx.Graph(); G.add_edge(1, 2, measurements_piv=[dict(tile_id=3, Q_kl=float("nan"), harm=dict(v=[np.nan, 1.0]))])
    G2 = pickle.loads(pickle.dumps(G))
    assert PT.is_empty(PT.diff(G, G2))
    G2.edges[1, 2]["measurements_piv"][0]["tile_id"] = 4
    assert len(PT.diff(G, G2)["changed_edges"]) == 1
