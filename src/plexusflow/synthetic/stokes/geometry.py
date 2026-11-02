"""Known-truth geometry: real Y-junctions from the measured graph and controlled single vessels, meshed (gmsh) and solved
for steady Stokes flow (scikit-fem).  Only needed to (re)build a benchmark bundle; running the benchmark reads the saved
solutions.

Junctions: carved interior degree-3 nodes with all three edges measured in one tile; each branch is a disk of its radius
swept along its (smoothed) centreline in the tile pixel frame, fused with a ball at the node, plus straight extensions
so the imposed inflow/outflow profiles sit away from the measured segments.  True lumen radius = RS × segmented radius
(RS = 1.2, calibrated from real lumen widths).
Single vessels: elliptical cross-sections lofted along straight / arc / S-bend / taper centrelines.
Units: pixels; flux per unit time; μ = 1 (velocity scales with flux only).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

RS = 1.2                                      # true lumen radius / segmented radius
H_DIV, TET_BUDGET = 4.0, 25000


def list_junctions(G, stage_root, n_max, seed=0):
    """Carved interior degree-3 junctions whose three edges are measured in one tile, radius 3–16 px, inside the tile by
    3R, each ≥ 3R long (truncated at 6R); picked at random (seeded), ≤ 3 per tile.  Paths in tile pixels, outward."""
    from ...config import StageConfig
    from ...analyze.geometry import load_tile_geometry, tile_video_shape, graph_to_tile_batch, graph_radius_to_tile_px
    from ...reconcile import measurement_table, carved_domain
    M = measurement_table(G); cfg = StageConfig.load(stage_root)
    shapes = {t: s for t in sorted({t for t, _ in M}) if (s := tile_video_shape(cfg, t)) is not None}
    geom = load_tile_geometry(cfg, video_shapes=shapes)
    dom = carved_domain(G); bnd = {dom["nodes"][i] for i in dom["bnd"]}; out = []
    for n in set(dom["nodes"]) - bnd:
        if G.degree(n) != 3:
            continue
        es = [(n, nb) if (n, nb) in G.edges else (nb, n) for nb in G.neighbors(n)]
        for t in shapes:
            if any((t, e) not in M for e in es):
                continue
            H, W = shapes[t]; br = []
            nxy = graph_to_tile_batch(np.array([[G.nodes[n]["x"], G.nodes[n]["y"]]]), geom, t)[0]
            for e in es:
                r = float(graph_radius_to_tile_px(float(G.edges[e]["radius"]), geom, t))
                p = graph_to_tile_batch(np.asarray(G.edges[e]["path"], float), geom, t)
                if np.linalg.norm(p[0] - nxy) > np.linalg.norm(p[-1] - nxy):
                    p = p[::-1]
                inside = (p[:, 0] > 3 * r) & (p[:, 0] < H - 3 * r) & (p[:, 1] > 3 * r) & (p[:, 1] < W - 3 * r)
                if not (3 <= r <= 16) or not inside[0]:
                    break
                p = p[:np.argmin(inside) if not inside.all() else len(p)]
                s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
                if s[-1] < 3 * r:
                    break
                p = p[s <= min(6 * r, s[-1] - r)]
                br.append(dict(edge=e, R_seg=r, path=p))
            else:
                out.append((n, t, br)); break
    rng = np.random.default_rng(seed); order = rng.permutation(len(out)); picked, per_tile = [], {}
    for i in order:                                              # spread over tiles (≤ 3 per tile)
        n, t, br = out[i]
        if per_tile.get(t, 0) < 3:
            picked.append(out[i]); per_tile[t] = per_tile.get(t, 0) + 1
        if len(picked) >= n_max:
            break
    print(f"geometric junction candidates {len(out)}; picked {len(picked)} over {len(per_tile)} tiles", flush=True)
    return picked


def prep_branches(br, rs=RS):
    """Smooth and resample each branch, true lumen radius R = rs × segmented, straight 4R extensions beyond the cut."""
    for b in br:
        p = b["path"]; p = np.column_stack([gaussian_filter1d(p[:, k], 2.0, mode="nearest") for k in (0, 1)])
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]; ns = np.linspace(0, s[-1], max(int(s[-1] / 2), 6))
        b["path"] = np.column_stack([np.interp(ns, s, p[:, 0]), np.interp(ns, s, p[:, 1])]); b["n_real"] = len(b["path"])
        b["R"] = rs * b["R_seg"]
        tdir = b["path"][-1] - b["path"][-4]; tdir /= np.linalg.norm(tdir)
        b["path"] = np.vstack([b["path"], b["path"][-1] + np.outer(np.arange(1, int(4 * b["R"] / 2) + 1) * 2.0, tdir)])
    return br


def build_junction_mesh(br, out_msh):
    """gmsh OpenCASCADE: disks swept along each branch spline + a ball at the node, fused; tet mesh h = R_min / H_DIV
    (coarsened to ≈ TET_BUDGET tets).  Physical groups: wall (10), cut plane of branch k (20 + k)."""
    import gmsh
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0); gmsh.model.add("junction")
    occ = gmsh.model.occ; vols = []
    node = np.mean([b["path"][0] for b in br], axis=0)
    for b in br:
        p = b["path"].copy(); p[0] = node                      # every branch starts exactly at the node
        pts = [occ.addPoint(x[1], -x[0], 0.0) for x in p]       # (x, y) = (col, −row): right-handed
        spl = occ.addSpline(pts); wire = occ.addWire([spl])
        d = np.array([p[1, 1] - p[0, 1], -(p[1, 0] - p[0, 0]), 0.0]); d /= np.linalg.norm(d)
        disk = occ.addDisk(node[1], -node[0], 0.0, b["R"], b["R"], zAxis=list(d))
        out = occ.addPipe([(2, disk)], wire, "DiscreteTrihedron"); vols += [x for x in out if x[0] == 3]
    ball = occ.addSphere(node[1], -node[0], 0.0, 1.05 * max(b["R"] for b in br)); vols.append((3, ball))
    fused, _ = occ.fuse([vols[0]], vols[1:]); occ.synchronize()
    vol = [v for v in fused if v[0] == 3]
    surfs = gmsh.model.getBoundary(vol, oriented=False)
    ends = []
    for b in br:
        e = b["path"][-1]; c_ = np.array([e[1], -e[0], 0.0]); bestf, dmin = None, 1e9
        for dim, tag in surfs:
            cm = np.array(occ.getCenterOfMass(dim, tag)); dd = np.linalg.norm(cm - c_)
            if dd < dmin:
                bestf, dmin = tag, dd
        ends.append(bestf)
    wall = [tag for dim, tag in surfs if tag not in ends]
    gmsh.model.addPhysicalGroup(3, [v[1] for v in vol], 1)
    gmsh.model.addPhysicalGroup(2, wall, 10, "wall")
    for k, tag in enumerate(ends):
        gmsh.model.addPhysicalGroup(2, [tag], 20 + k, f"end{k}")
    # element size: R_min / H_DIV, coarsened if the lumen would need more than ~TET_BUDGET tets (direct-solver memory)
    vol = sum(occ.getMass(3, v[1]) for v in vol)
    h = max(min(b["R"] for b in br) / H_DIV, (6 * np.sqrt(2) * vol / TET_BUDGET) ** (1 / 3))
    gmsh.option.setNumber("Mesh.MeshSizeMax", h); gmsh.option.setNumber("Mesh.MeshSizeMin", h / 2)
    gmsh.model.mesh.generate(3); gmsh.write(str(out_msh)); gmsh.finalize()
    return node


def solve_junction_basis(msh_path, br):
    """Steady Stokes (Taylor–Hood P2–P1), no-slip wall, fully developed Poiseuille on each cut plane.  Two basis solutions
    with (Q0, Q1, Q2) = (1, 1, 0) and (1, 0, 1), branch 0 the parent inflow; any split is their combination."""
    import meshio
    from skfem import MeshTet, ElementVector, ElementTetP2, ElementTetP1, Basis, BilinearForm, asm, bmat, condense, solve
    from skfem.helpers import grad, div, ddot
    m = meshio.read(msh_path)
    tets = m.cells_dict["tetra"]; P = m.points
    mesh = MeshTet(P.T, tets.T)
    # boundary facets by physical tag (triangles)
    tri = m.cells_dict["triangle"]; tag = m.cell_data_dict["gmsh:physical"]["triangle"]
    fkey = {tuple(sorted(f)): i for i, f in enumerate(mesh.facets.T)}
    groups = {}
    for f, g in zip(tri, tag):
        i = fkey.get(tuple(sorted(f)))
        if i is not None:
            groups.setdefault(int(g), []).append(i)
    eu, ep = ElementVector(ElementTetP2()), ElementTetP1()
    ub = Basis(mesh, eu, intorder=4); pb = ub.with_element(ep)
    @BilinearForm
    def a(u, v, w):
        return ddot(grad(u), grad(v))
    @BilinearForm
    def b(u, q, w):
        return -div(u) * q
    A = asm(a, ub); B = asm(b, ub, pb)
    K = bmat([[A, B.T], [B, 1e-10 * asm(BilinearForm(lambda p, q, w: p * q), pb)]], "csr")
    nu = ub.N
    # boundary DOFs: all velocity DOFs on wall + ends (Dirichlet everywhere on the boundary)
    allb = np.concatenate([np.asarray(v) for v in groups.values()])
    D = ub.get_dofs(np.unique(allb)).all()
    xd = ub.doflocs                                        # (3, N) locations of velocity DOFs
    ends = []
    for k, b_ in enumerate(br):
        fac = np.asarray(groups[20 + k])
        c_ = np.array([b_["path"][-1][1], -b_["path"][-1][0], 0.0])
        t_ = np.array([b_["path"][-1][1] - b_["path"][-2][1], -(b_["path"][-1][0] - b_["path"][-2][0]), 0.0]); t_ /= np.linalg.norm(t_)
        ends.append((fac, c_, t_, b_["R"]))
    out = {}
    for name, Qs in (("A", (1.0, 1.0, 0.0)), ("B", (1.0, 0.0, 1.0))):
        x = np.zeros(K.shape[0])
        for k, (fac, c_, t_, R) in enumerate(ends):
            sgn = -1.0 if k == 0 else 1.0                      # branch 0: inflow (−t at its far end points into the domain)
            dv = ub.get_dofs(fac)
            for ci, key in enumerate(("u^1", "u^2", "u^3")):
                dk = dv.all(key); r = xd[:, dk] - c_[:, None]; rr2 = np.sum(r ** 2, 0) - (t_ @ r) ** 2
                x[dk] = sgn * 2 * Qs[k] / (np.pi * R ** 2) * np.clip(1 - rr2 / R ** 2, 0, None) * t_[ci]
        sol = solve(*condense(K, np.zeros(K.shape[0]), x=x, D=D))
        out[name] = sol[:nu]
    return mesh, ub, out, groups


def centreline(kind, a, L_meas=10.0, ext=4.0, kappaR=4.0):
    """Vessel centreline: (N, 2) points (x, y) at ~0.5a spacing, measured region flagged; total = ext·a + L_meas·a + ext·a."""
    Lm, Le = L_meas * a, ext * a; s = np.arange(0, Lm + 2 * Le + 1e-9, 0.5 * a)
    if kind in ("straight", "taper"):
        P = np.column_stack([s, np.zeros_like(s)])
    elif kind == "arc":
        Rc = kappaR * a; th = np.clip(s - Le, 0, Lm) / Rc
        P = np.zeros((len(s), 2)); body = (s >= Le) & (s <= Le + Lm)
        P[body] = np.column_stack([Le + Rc * np.sin(th[body]), Rc * (1 - np.cos(th[body]))])
        pre = s < Le; P[pre] = np.column_stack([s[pre], 0 * s[pre]])
        post = s > Le + Lm; thE = Lm / Rc; e = np.array([Le + Rc * np.sin(thE), Rc * (1 - np.cos(thE))]); d = np.array([np.cos(thE), np.sin(thE)])
        P[post] = e + np.outer(s[post] - Le - Lm, d)
    elif kind == "sbend":
        amp = 1.5 * a; x = s; y = np.where((s >= Le) & (s <= Le + Lm), amp * np.sin(2 * np.pi * (s - Le) / Lm), 0.0)
        P = np.column_stack([x, y])
    meas = (s >= Le) & (s <= Le + Lm)
    return P, meas, s


def build_vessel(out_dir, kind, a0, eps, taper=0.7, name=None, pillar=0.0):
    """Single vessel: elliptical sections (in-plane a, depth ε·a) lofted along the centreline, 4a extensions; Stokes with
    elliptical Poiseuille caps, Q = 1.  Writes <out_dir>/<name>.{msh,npz,pkl}; returns (name, number of tets)."""
    import gmsh
    P, meas, s = centreline(kind, a0)
    Le = 4.0 * a0; Lm = 10.0 * a0
    afun = (lambda ss: a0 * np.where(ss < Le, 1.0, np.where(ss > Le + Lm, taper, 1 + (taper - 1) * (ss - Le) / Lm))) if kind == "taper" else (lambda ss: a0 + 0 * ss)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0); gmsh.model.add("v"); occ = gmsh.model.occ
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True); wires = []
    for p, t, ss in zip(P, T, s):
        a = float(afun(ss)); b = eps * a; t3 = [t[0], t[1], 0.0]; n_in = [-t[1], t[0], 0.0]
        if a >= b:
            e = occ.addEllipse(p[0], p[1], 0.0, a, b, zAxis=t3, xAxis=n_in)
        else:
            e = occ.addEllipse(p[0], p[1], 0.0, b, a, zAxis=t3, xAxis=[0, 0, 1.0])
        wires.append(occ.addCurveLoop([e]))
    vol = occ.addThruSections(wires, makeSolid=True, makeRuled=False)
    if pillar > 0:                                       # tissue post through the full depth at the middle of the measured region
        im = int(np.argmin(np.abs(s - (4.0 * a0 + 5.0 * a0)))); pc = P[im]
        cyl = occ.addCylinder(pc[0], pc[1], -2 * a0, 0, 0, 4 * a0, pillar)
        vol, _ = occ.cut(vol, [(3, cyl)])
    occ.synchronize()
    v3 = [x for x in vol if x[0] == 3]; surfs = gmsh.model.getBoundary(v3, oriented=False); ends = []
    for q in (P[0], P[-1]):
        c_ = np.array([q[0], q[1], 0.0]); ends.append(min(surfs, key=lambda d: np.linalg.norm(np.array(occ.getCenterOfMass(*d)) - c_))[1])
    wall = [tg for dm, tg in surfs if tg not in ends]
    gmsh.model.addPhysicalGroup(3, [x[1] for x in v3], 1); gmsh.model.addPhysicalGroup(2, wall, 10)
    gmsh.model.addPhysicalGroup(2, [ends[0]], 20); gmsh.model.addPhysicalGroup(2, [ends[1]], 21)
    volume = sum(occ.getMass(3, x[1]) for x in v3)
    h = max(min(a0 * eps, a0) / 3.0, (6 * np.sqrt(2) * volume / 25000) ** (1 / 3))
    gmsh.option.setNumber("Mesh.MeshSizeMax", h); gmsh.option.setNumber("Mesh.MeshSizeMin", h / 2)
    name = name or f"{kind}_a{a0:g}_e{eps:g}" + (f"_p{pillar:g}" if pillar > 0 else ""); out_dir = Path(out_dir); msh = out_dir / f"{name}.msh"
    gmsh.model.mesh.generate(3); gmsh.write(str(msh)); gmsh.finalize()
    # Stokes
    import meshio
    from skfem import MeshTet, ElementVector, ElementTetP2, ElementTetP1, Basis, BilinearForm, asm, bmat, condense, solve
    from skfem.helpers import grad, div, ddot
    m = meshio.read(msh); mesh = MeshTet(m.points.T, m.cells_dict["tetra"].T)
    fkey = {tuple(sorted(f)): i for i, f in enumerate(mesh.facets.T)}; groups = {}
    for f, g in zip(m.cells_dict["triangle"], m.cell_data_dict["gmsh:physical"]["triangle"]):
        i = fkey.get(tuple(sorted(f)))
        if i is not None:
            groups.setdefault(int(g), []).append(i)
    ub = Basis(mesh, ElementVector(ElementTetP2()), intorder=4); pb = ub.with_element(ElementTetP1())
    A = asm(BilinearForm(lambda u, v, w: ddot(grad(u), grad(v))), ub); B = asm(BilinearForm(lambda u, q, w: -div(u) * q), ub, pb)
    K = bmat([[A, B.T], [B, 1e-10 * asm(BilinearForm(lambda p, q, w: p * q), pb)]], "csr")
    D = ub.get_dofs(np.unique(np.concatenate([np.asarray(v) for v in groups.values()]))).all(); x = np.zeros(K.shape[0]); xd = ub.doflocs
    for g, q, i in ((20, P[0], 0), (21, P[-1], -1)):
        t = T[i]; t3 = np.array([t[0], t[1], 0.0]); n3 = np.array([-t[1], t[0], 0.0]); a = float(afun(s[i])); b = eps * a
        dv = ub.get_dofs(np.asarray(groups[g]))
        for ci, key in enumerate(("u^1", "u^2", "u^3")):
            dk = dv.all(key); r = xd[:, dk] - np.array([q[0], q[1], 0.0])[:, None]
            yy, zz = n3 @ r, r[2]; x[dk] = 2.0 / (np.pi * a * b) * np.clip(1 - yy ** 2 / a ** 2 - zz ** 2 / b ** 2, 0, None) * t3[ci]
    sol = solve(*condense(K, np.zeros(K.shape[0]), x=x, D=D))[:ub.N]
    np.savez_compressed(out_dir / f"{name}.npz", p=mesh.p, t=mesh.t, u=sol)
    meta = dict(kind=kind, a0=a0, eps=eps, taper=taper if kind == "taper" else 1.0, P=P, meas=meas, s=s, a_meas=float(np.mean(afun(s[meas]))),
                caps=[(np.array([P[0][0], P[0][1], 0.0]), -np.array([T[0][0], T[0][1], 0.0]), float(afun(s[0])), eps * float(afun(s[0])), -1.0),
                      (np.array([P[-1][0], P[-1][1], 0.0]), np.array([T[-1][0], T[-1][1], 0.0]), float(afun(s[-1])), eps * float(afun(s[-1])), 1.0)])
    (out_dir / f"{name}.pkl").write_bytes(pickle.dumps(meta))
    return name, len(m.cells_dict["tetra"])
