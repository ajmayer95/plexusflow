"""Local refinement loop: stage a candidate graph (+ mask), measure only its new vessels, refit, compare, accept.

    stage → evaluate (background job, project venv) → review in the workbench → accept (patch log) or discard

A candidate lives in ``graph/patches/pending/<stamp>/``:
    candidate.gpickle   the spliced graph (later: with the new vessels measured)
    mask.tif            the candidate mask (only when the mask was edited)
    meta.json           kind, note, polygon, base graph size + mtime, splice report
    result.json / .pkl  evaluation (written by ``evaluate``)
    job.log             the job's output

``evaluate`` (CLI: ``plexusflow refine-eval STAGE PENDING``):
1. measures the new vessels: ``analyze_stage(local=True, skip_analyzed=True)`` on the tiles that recorded vessels near the
   region, writing only into the candidate (the stage graph, provenance and other tiles are untouched);
2. refits the DC field for the current stage graph and for the candidate with the SAME reliability model
   (results/reliability_model_v1.json) and the production quasi-likelihood solver (single start, no multistart);
3. compares, globally (inflow, L share, outflow per side, large reversals) and locally within ``pad`` px of the region:
   fit-free same-tile junction balance of the raw measurements (the main local criterion), measured vessels, reversals.
``accept`` writes the candidate through the patch log, refusing if the stage graph changed since staging.
"""
from __future__ import annotations

import datetime
import json
import pickle
import shutil
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import networkx as nx

PAD = 300.0                      # px around the region for local metrics / tile selection
REVERSAL_Q = 0.5                 # nL/s: |Q'| above this against the measured direction = large reversal


# ── staging ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def pending_root(graph_path: Path) -> Path:
    return Path(graph_path).parent / "patches" / "pending"


def stage_candidate(graph_path: Path, G_new: nx.Graph, *, polygon, kind: str, note: str = "", report: Optional[dict] = None,
                    mask_new: Optional[np.ndarray] = None, dump: Optional[Callable] = None) -> Path:
    """Write a candidate to ``graph/patches/pending/<stamp>/``.  Returns the directory.  Writes nothing else."""
    graph_path = Path(graph_path); st = graph_path.stat()
    d = pending_root(graph_path) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S"); d.mkdir(parents=True, exist_ok=False)
    with open(d / "candidate.gpickle", "wb") as fh:
        (dump or (lambda f: pickle.dump(G_new, f, protocol=4)))(fh)
    if mask_new is not None:
        import tifffile
        tifffile.imwrite(d / "mask.tif", (np.asarray(mask_new, bool) * 255).astype(np.uint8), compression="zlib")
    meta = dict(kind=kind, note=note, polygon=np.asarray(polygon, float).round(2).tolist(), graph=str(graph_path),
                base_size=st.st_size, base_mtime=st.st_mtime, mask_edited=mask_new is not None,
                report={k: v for k, v in (report or {}).items() if k != "polygon"})
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=_jsonable))
    return d


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def list_pending(graph_path: Path) -> list:
    r = pending_root(graph_path)
    return sorted(p for p in r.iterdir() if (p / "meta.json").exists()) if r.exists() else []


def discard(pending: Path) -> None:
    pending = Path(pending)
    if pending.parent.name != "pending":
        raise ValueError(f"not a pending candidate: {pending}")
    shutil.rmtree(pending)


# ── geometry helpers ─────────────────────────────────────────────────────────────────────────────────────────────────

def _poly_points(poly, step=5.0):
    P = np.asarray(poly, float); out = []
    for a, b in zip(P, np.roll(P, -1, 0)):
        n = max(int(np.linalg.norm(b - a) / step), 1); out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    return np.vstack(out)


def near_region(G, poly, pad=PAD):
    """Edges whose centreline midpoint is inside the polygon or within ``pad`` px of its border."""
    from matplotlib.path import Path as MplPath
    from scipy.spatial import cKDTree
    T = cKDTree(_poly_points(poly)); reg = MplPath(np.asarray(poly, float)); out = set()
    for u, v, d in G.edges(data=True):
        p = np.asarray(d["path"], float); m = p[len(p) // 2]
        if reg.contains_point(m) or T.query(m)[0] <= pad:
            out.add((u, v))
    return out


def tiles_for_region(G, poly, pad=PAD) -> list:
    return sorted({int(m["tile_id"]) for e in near_region(G, poly, pad) for m in (G.edges[e].get("measurements_piv") or [])})


# ── fit + metrics ────────────────────────────────────────────────────────────────────────────────────────────────────

def quick_fit(G, model) -> dict:
    """Production DC estimator, single start: vessel inputs → uniform projection → quasi-likelihood (plexusflow.reconcile)."""
    from ..reconcile import measurement_table
    from ..reconcile.reliability import c_of
    from ..reconcile.dc import vessel_input, ql_solve
    from ..reconcile.network import carved_domain, project, emergent, EPS_UNIMAGED
    dom = carved_domain(G); es = [tuple(e) for e in dom["edges"]]; ix = {e: i for i, e in enumerate(es)}; me = len(es)
    M = measurement_table(G); by = {}
    for (t, e), m in M.items():
        if e in ix:
            by.setdefault(e, []).append((m, c_of(model, m)))
    y = np.zeros(me); C = np.ones(me); im = np.zeros(me, bool); sb = np.zeros(me)
    for e, L in by.items():
        i = ix[e]; y[i], C[i], sb[i] = vessel_input(L); im[i] = True
    s0 = 0.01 * np.median(np.abs(y[im])); s0v = np.sqrt(s0 ** 2 + sb ** 2); Cw = np.where(im, C, 1.0)
    Q0 = project(y, np.where(im, 1.0, EPS_UNIMAGED), dom); mm = float(np.median(1 / (Cw[im] ** 2 * Q0[im] ** 2 + s0 ** 2)))
    Q, info = ql_solve(Q0, y, Cw, im, s0v, mm, dom)
    fb = emergent(Q, dom); inj = fb[dom["inj"]]; out = fb[dom["out"]]; so = np.asarray(dom["side_out"])
    rev = [es[i] for i in np.flatnonzero(im & (np.sign(Q) != np.sign(y)) & (np.abs(Q) > REVERSAL_Q))]
    return dict(edges=es, Q=Q, y=np.where(im, y, np.nan), c=np.where(im, C, np.nan), imaged=im, M=M,
                inflow=float(inj.sum()), left=float(inj[np.asarray(dom["side"]) == "L"].sum() / max(inj.sum(), 1e-12)),
                out_L=float(out[so == "L"].sum()), out_R=float(out[so == "R"].sum()), reversals=rev, converged=bool(info.get("converged", True)) if isinstance(info, dict) else True)


def junction_balance(G, M, model, edges) -> dict:
    """Fit-free check of the raw measurements: at same-tile junctions whose incident vessels were all measured on that
    tile, imbalance = |Σ signed Q| / Σ|Q|.  Restricted to junctions with an incident edge in ``edges``."""
    from ..reconcile.reliability import same_tile_junctions
    edges = {frozenset(e) for e in edges}; vals = []
    for t in sorted({t for t, _ in M}):
        meas = {e for (tt, e) in M if tt == t}
        for J in same_tile_junctions(G, t, meas):
            if not any(frozenset(e) in edges for e, _ in J):
                continue
            q = np.array([s * M[(t, e)]["Q"] for e, s in J]); den = np.abs(q).sum()
            if den > 0:
                vals.append(float(abs(q.sum()) / den))
    v = np.array(vals)
    return dict(n=len(v), median=float(np.median(v)) if len(v) else None, gt_half=float(np.mean(v > 0.5)) if len(v) else None)


def _local(G, F, region_edges) -> dict:
    ix = {e: i for i, e in enumerate(F["edges"])}; idx = [ix[e] for e in region_edges if e in ix]
    im = F["imaged"][idx]; rev = [e for e in F["reversals"] if e in set(region_edges)]
    return dict(vessels=len(idx), measured=int(im.sum()), reversals=[[int(u), int(v)] for u, v in rev], n_reversals=len(rev))


# ── evaluate ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def evaluate(stage_root: Path, pending: Path, *, measure: bool = True, workers: Optional[int] = None, log=print) -> dict:
    """Measure the candidate's new vessels, refit current vs candidate, write result.json / result.pkl."""
    import plexusflow.paths as paths
    from ..config import StageConfig
    from .patches import require_numpy1
    if measure:
        require_numpy1()                      # the measurement rewrites the candidate pickle
    pending = Path(pending); meta = json.loads((pending / "meta.json").read_text())
    cfg = StageConfig.load(Path(stage_root)); graph_path = Path(meta["graph"]); cand = pending / "candidate.gpickle"
    G0 = pickle.loads(graph_path.read_bytes()); poly = np.asarray(meta["polygon"], float)
    tiles = tiles_for_region(G0, poly)
    if measure:
        Gc = pickle.loads(cand.read_bytes())
        todo = [e for e in Gc.edges if Gc.edges[e].get("spliced") and not Gc.edges[e].get("measurements_piv")]
        log(f"[refine] measuring {len(todo)} new vessels on tiles {tiles}")
        if todo and tiles:
            from ..analyze import analyze_stage
            analyze_stage(cfg, tile_ids=tiles, skip_analyzed=True, out_path=cand, n_workers=workers, local=True,
                          mask_path=(pending / "mask.tif") if meta.get("mask_edited") else None)
    G1 = pickle.loads(cand.read_bytes())
    new = [e for e in G1.edges if G1.edges[e].get("spliced")]
    n_meas = sum(1 for e in new if any((m.get("quality") or {}).get("kl_ok") for m in G1.edges[e].get("measurements_piv") or []))
    model = json.loads((paths.RESULTS / "reliability_model_v1.json").read_text())
    log("[refine] fitting current graph …"); F0 = quick_fit(G0, model)
    log("[refine] fitting candidate …"); F1 = quick_fit(G1, model)
    r0, r1 = near_region(G0, poly), near_region(G1, poly)
    res = dict(
        tiles=tiles, new_vessels=len(new), new_measured_kl_ok=n_meas,
        current=dict(inflow=F0["inflow"], left=F0["left"], out_L=F0["out_L"], out_R=F0["out_R"], n_reversals=len(F0["reversals"]),
                     local=_local(G0, F0, r0), junctions=junction_balance(G0, F0["M"], model, r0)),
        candidate=dict(inflow=F1["inflow"], left=F1["left"], out_L=F1["out_L"], out_R=F1["out_R"], n_reversals=len(F1["reversals"]),
                       local=_local(G1, F1, r1), junctions=junction_balance(G1, F1["M"], model, r1)),
        stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    (pending / "result.json").write_text(json.dumps(res, indent=1, default=_jsonable))
    with open(pending / "result.pkl", "wb") as fh:
        pickle.dump(dict(current={k: F0[k] for k in ("edges", "Q", "y", "c", "imaged", "reversals")},
                         candidate={k: F1[k] for k in ("edges", "Q", "y", "c", "imaged", "reversals")}), fh, protocol=4)
    log("[refine] " + format_result(res))
    return res


def format_result(res: dict) -> str:
    a, b = res["current"], res["candidate"]
    j = lambda x: (f"{x['junctions']['median']:.2f} (n {x['junctions']['n']}, >0.5: {x['junctions']['gt_half']:.0%})"
                   if x["junctions"]["n"] else "none")
    return "\n".join([
        f"new vessels {res['new_vessels']} (measured, Kloosterman-valid: {res['new_measured_kl_ok']}); tiles {res['tiles']}",
        f"                      current      candidate",
        f"inflow (nL/s)         {a['inflow']:8.2f}    {b['inflow']:8.2f}",
        f"left share            {a['left']:8.1%}    {b['left']:8.1%}",
        f"outflow L / R         {a['out_L']:+.2f}/{a['out_R']:+.2f}  {b['out_L']:+.2f}/{b['out_R']:+.2f}",
        f"large reversals       {a['n_reversals']:8d}    {b['n_reversals']:8d}",
        f"near region: vessels  {a['local']['vessels']:8d}    {b['local']['vessels']:8d}",
        f"  measured            {a['local']['measured']:8d}    {b['local']['measured']:8d}",
        f"  reversals           {a['local']['n_reversals']:8d}    {b['local']['n_reversals']:8d}",
        f"  junction imbalance (raw, fit-free)  current {j(a)}  candidate {j(b)}"])


# ── accept ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def accept(pending: Path, *, mask_path: Optional[Path] = None, note: Optional[str] = None, G_current: Optional[nx.Graph] = None,
           dump: Optional[Callable] = None) -> dict:
    """Write the candidate to the stage through the patch log.  Refuses if the stage graph changed since staging."""
    from .patches import PatchLog
    pending = Path(pending); meta = json.loads((pending / "meta.json").read_text()); gp = Path(meta["graph"]); st = gp.stat()
    if st.st_size != meta["base_size"] or abs(st.st_mtime - meta["base_mtime"]) > 1e-3:
        raise RuntimeError("the stage graph changed since this candidate was staged — re-run the splice on the current graph")
    G0 = G_current if G_current is not None else pickle.loads(gp.read_bytes())
    G1 = pickle.loads((pending / "candidate.gpickle").read_bytes())
    mb = ma = None
    if meta.get("mask_edited"):
        import tifffile
        if mask_path is None:
            raise ValueError("candidate edits the mask: pass mask_path")
        mb = tifffile.imread(mask_path) > 127; ma = tifffile.imread(pending / "mask.tif") > 127
    res = json.loads((pending / "result.json").read_text()) if (pending / "result.json").exists() else {}
    rep = dict(meta.get("report") or {}); rep.update({f"eval_{k}": v for k, v in res.items() if isinstance(v, (int, float, str))})
    log = PatchLog(gp, mask_path)
    entry = log.accept(G0, G1, kind=meta["kind"], note=note if note is not None else meta.get("note", ""), polygon=meta["polygon"],
                       report=rep, mask_before=mb, mask_after=ma, dump=dump)
    shutil.copy2(pending / "meta.json", log.dir / f"{entry['id']}_meta.json")
    if (pending / "result.json").exists():
        shutil.copy2(pending / "result.json", log.dir / f"{entry['id']}_result.json")
    discard(pending)
    return entry
