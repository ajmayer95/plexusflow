"""Anatomical radius R(s) from the mask (reconcile.radius) and the Kloosterman section rule (no flow-rate band)."""
import numpy as np

from plexusflow.analyze.kloosterman import kloosterman_scalar, PRODUCTION_KW
from plexusflow.reconcile.radius import mask_edt, boundary_profile, summarize


def _arc_tube(R=6.0, Rc=40.0, H=140, W=140):
    """Mask of a curved tube (quarter annulus of centre-line radius Rc), and its graph path as a COARSE polyline
    (3 nodes — the chords cut ~3 px toward the inner wall), in graph (x, y) with row = H − y."""
    yy, xx = np.mgrid[:H, :W]; cy, cx = 20.0, 20.0
    d = np.hypot(yy - cy, xx - cx); ang = np.arctan2(yy - cy, xx - cx)
    m = (np.abs(d - Rc) <= R) & (ang >= 0) & (ang <= np.pi / 2)
    th = np.linspace(0, np.pi / 2, 3); rows, cols = cy + Rc * np.sin(th), cx + Rc * np.cos(th)
    return m, np.column_stack([cols, H - rows])


def test_medial_radius_recovers_tube_radius_on_coarse_curved_path():
    m, path = _arc_tube(R=6.0)
    edt = mask_edt(m)
    r_path = boundary_profile(edt, path, search=0)        # EDT on the chord only: reads low between nodes
    r_med = boundary_profile(edt, path)
    r_true = edt.max() - 0.5                                   # mask boundary of the discretised tube
    assert np.median(r_path) < r_true - 1.0
    assert abs(np.median(r_med) - r_true) < 0.5
    s = summarize(r_med, offset_px=3.0)
    assert abs(s["median"] - np.median(r_med) - 3.0) < 1e-9 and 0.9 < s["taper"] < 1.05


def test_sections_outside_old_band_are_kept_but_reversed_ones_are_not():
    H, W, R, v0 = 60, 220, 6.0, 1.0
    yy, xx = np.mgrid[:H, :W].astype(float); rho = yy - 30
    vx = np.where(np.abs(rho) <= R, v0 * (1 - (rho / R) ** 2), 0.0)
    vx[:, 120:150] *= 2.2                                       # a fast stretch: > 1.5× the median section flux
    flow = np.stack([vx, np.zeros_like(vx)], axis=-1)
    cl = np.column_stack([np.full(170, 30.0), np.arange(25, 195, dtype=float)])
    new = kloosterman_scalar(flow, cl, R, **PRODUCTION_KW)
    old = kloosterman_scalar(flow, cl, R, **{**PRODUCTION_KW, "outlier_mode": "median_band", "median_lo": 0.5, "median_hi": 1.5})
    assert new["n_valid"] > old["n_valid"] and new["Q_char"] > old["Q_char"]
    flow2 = flow.copy(); flow2[:, 60:80, 0] *= -1                # a reversed stretch is still rejected
    rev = kloosterman_scalar(flow2, cl, R, **PRODUCTION_KW)
    assert any(r == "sign_flip" for r in rev["per_slice_reason"]) and rev["Q_char"] > 0


def test_contradictory_sections_give_no_flux():
    # Fewer than min_valid_slices sign-consistent sections (here one forward and one reversed fit): no flux, instead of
    # the pre-2026-11-11 fallback that averaged every finite section regardless of sign.
    H, W, R = 60, 220, 6.0
    yy, xx = np.mgrid[:H, :W].astype(float); rho = yy - 30
    vx = np.where(np.abs(rho) <= R, 1 - (rho / R) ** 2, 0.0)
    flow = np.stack([vx, np.zeros_like(vx)], axis=-1); flow[:, 102:, 0] *= -1
    cl = np.column_stack([np.full(4, 30.0), np.arange(100, 104, dtype=float)])
    r = kloosterman_scalar(flow, cl, R, **PRODUCTION_KW)
    assert r["reason"] == "too_few_consistent_slices" and np.isnan(r["Q_char"]) and r["Q_char_valid_frac"] == 0.0
    assert "ok" not in list(r["per_slice_reason"])


def _straight_tube_with_side_branch(a=6, H=120, W=200):
    """Horizontal tube of half-width a (rows 60 ± a) with a wide perpendicular branch at columns 95–125; path along row 60."""
    m = np.zeros((H, W), bool); m[60 - a:60 + a + 1, 10:190] = True; m[:60, 95:126] = True
    path = np.column_stack([np.arange(20, 181, 10, dtype=float), np.full(17, H - 60.0)])
    return m, path


def test_tube_interior_profile_excludes_junction_and_recovers_half_width():
    from plexusflow.reconcile.radius import tube_interior_profile
    m, path = _straight_tube_with_side_branch()
    a, wl = tube_interior_profile(m, path)
    cols = np.linspace(20, 180, len(a))
    in_branch = (cols > 97) & (cols < 123)
    assert not wl[in_branch].any()                               # the normal runs up the branch: not a tube section
    assert wl[~in_branch].mean() > 0.9
    assert abs(np.median(a[wl]) - 6.5) < 0.6                    # half-width to the first background pixel (6 px + ½ step)


def test_tube_radii_resistance_compliance_and_fallback():
    import networkx as nx
    from plexusflow.reconcile.radius import tube_radii
    m, path = _straight_tube_with_side_branch()
    m[60 - 9:60 + 10, 150:181] = True                            # a wider stretch: a_C > median
    G = nx.Graph(); G.add_edge(0, 1, path=path, radius=9.0); G.add_edge(2, 3, path=path[:1], radius=4.0)
    T = tube_radii(G, [(0, 1), (2, 3)], m)
    assert T["profiled"].tolist() == [True, False]
    assert T["compliance"][0] > T["resistance"][0] and abs(T["resistance"][0] - 6.5) < 0.6
    assert T["resistance"][1] == T["compliance"][1] == 4.0      # unprofiled: stored radius for both
