"""Patch log for the stage graph: every accepted refinement (consolidation, region re-vectorisation, …) is stored as a
reversible diff, so the working graph = base + patches in order, and any patch can be reverted.

Layout (``graph/patches/``):
    log.json            list of entries, oldest first: id, kind, note, stamp, status (applied / reverted), polygon,
                        counts, mask change (bbox), graph file and its size/mtime after the patch
    base_<stamp>.gpickle  copy of the stage graph when the log was started (written once)
    <id>.pkl.gz         the diff: removed / added / changed nodes and edges with FULL attributes, plus the mask crop
                        before/after (packed bits) when the patch changed the mask

A diff is computed between two graphs (``diff``); ``apply`` / ``revert`` replay it exactly.  Reverting a patch that a
later applied patch builds on (shares nodes or edges) is refused.  Writes to the stage go through ``PatchLog.accept`` /
``PatchLog.revert``, which write the graph atomically (temp file + rename) and record mask edits in the mask journal
(plexusflow.mask.journal) as well.
"""
from __future__ import annotations

import datetime
import gzip
import json
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx


def _same(a, b) -> bool:
    """Deep equality with NaN == NaN (measurement records carry NaNs, e.g. Q_kl of a failed fit)."""
    if a is b:
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape:
                return False
            if a.dtype == object or b.dtype == object:
                return all(_same(x, y) for x, y in zip(a.ravel(), b.ravel()))
            return np.array_equal(a, b, equal_nan=a.dtype.kind in "fc" and b.dtype.kind in "fc")
        except Exception:
            return False
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return a == b or (a != a and b != b)
    try:
        r = a == b
        return bool(r) if not isinstance(r, np.ndarray) else bool(r.all())
    except Exception:
        return False


def _attrs_equal(a: dict, b: dict) -> bool:
    if a is b:
        return True
    if a.keys() != b.keys():
        return False
    return all(_same(a[k], b[k]) for k in a)


def _ekey(G, u, v):
    return (u, v) if G.has_edge(u, v) else (v, u)


def diff(G_old: nx.Graph, G_new: nx.Graph) -> dict:
    """Node / edge differences between two graphs, with full attribute copies (shallow per attribute)."""
    on, nn = set(G_old.nodes), set(G_new.nodes)
    oe = {frozenset(e): e for e in G_old.edges}; ne = {frozenset(e): e for e in G_new.edges}
    d = dict(removed_nodes={n: dict(G_old.nodes[n]) for n in on - nn},
             added_nodes={n: dict(G_new.nodes[n]) for n in nn - on},
             removed_edges={oe[k]: dict(G_old.edges[oe[k]]) for k in oe.keys() - ne.keys()},
             added_edges={ne[k]: dict(G_new.edges[ne[k]]) for k in ne.keys() - oe.keys()},
             changed_nodes={}, changed_edges={})
    for n in on & nn:
        a, b = G_old.nodes[n], G_new.nodes[n]
        if not _attrs_equal(a, b):
            d["changed_nodes"][n] = (dict(a), dict(b))
    for k in oe.keys() & ne.keys():
        a, b = G_old.edges[oe[k]], G_new.edges[ne[k]]
        if not _attrs_equal(a, b):
            d["changed_edges"][oe[k]] = (dict(a), dict(b))
    return d


def is_empty(d: dict) -> bool:
    return not any(d[k] for k in ("removed_nodes", "added_nodes", "removed_edges", "added_edges", "changed_nodes", "changed_edges"))


def apply(G: nx.Graph, d: dict) -> None:
    """Apply a diff to G in place."""
    G.remove_edges_from([e for e in d["removed_edges"] if G.has_edge(*e)])
    G.remove_nodes_from([n for n in d["removed_nodes"] if n in G])
    for n, a in d["added_nodes"].items():
        G.add_node(n, **a)
    for n, (_, b) in d["changed_nodes"].items():
        G.nodes[n].clear(); G.nodes[n].update(b)
    for (u, v), a in d["added_edges"].items():
        G.add_edge(u, v, **a)
    for (u, v), (_, b) in d["changed_edges"].items():
        e = G.edges[u, v]; e.clear(); e.update(b)


def invert(d: dict) -> dict:
    return dict(removed_nodes=d["added_nodes"], added_nodes=d["removed_nodes"], removed_edges=d["added_edges"],
                added_edges=d["removed_edges"], changed_nodes={n: (b, a) for n, (a, b) in d["changed_nodes"].items()},
                changed_edges={e: (b, a) for e, (a, b) in d["changed_edges"].items()})


def revert(G: nx.Graph, d: dict) -> None:
    apply(G, invert(d))


def touched(d: dict) -> tuple:
    """(nodes, edges as frozensets) a diff touches — used for revert conflict checks."""
    nodes = set(d["removed_nodes"]) | set(d["added_nodes"]) | set(d["changed_nodes"])
    edges = {frozenset(e) for k in ("removed_edges", "added_edges", "changed_edges") for e in d[k]}
    for k in ("removed_edges", "added_edges"):
        for u, v in d[k]:
            nodes |= {u, v}
    return nodes, edges


def summary(d: dict) -> dict:
    return {k: len(d[k]) for k in ("removed_nodes", "added_nodes", "removed_edges", "added_edges", "changed_nodes", "changed_edges")}


class PatchLog:
    """The stage's patch log (``graph/patches``) for one graph file."""

    def __init__(self, graph_path: Path, mask_path: Optional[Path] = None):
        self.graph_path = Path(graph_path)
        self.mask_path = Path(mask_path) if mask_path else None
        self.dir = self.graph_path.parent / "patches"

    # ── log ──
    @property
    def log_path(self) -> Path:
        return self.dir / "log.json"

    def entries(self) -> list:
        return json.loads(self.log_path.read_text()) if self.log_path.exists() else []

    def _write_log(self, entries: list) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.log_path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(entries, indent=1)); os.replace(tmp, self.log_path)

    def load_diff(self, pid: str) -> dict:
        with gzip.open(self.dir / f"{pid}.pkl.gz", "rb") as fh:
            return pickle.load(fh)

    def _next_id(self) -> str:
        return f"{len(self.entries()) + 1:04d}"

    # ── graph I/O ──
    def _write_graph(self, G: nx.Graph, dump=None) -> None:
        fd, tmp = tempfile.mkstemp(prefix=".patch_", suffix=".gpickle", dir=str(self.graph_path.parent)); os.close(fd)
        try:
            with open(tmp, "wb") as fh:
                (dump or (lambda f: pickle.dump(G, f, protocol=4)))(fh)
            os.replace(tmp, self.graph_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _ensure_base(self) -> None:
        if not any(self.dir.glob("base_*.gpickle")) and self.graph_path.exists():
            self.dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.graph_path, self.dir / f"base_{datetime.datetime.now():%Y%m%d_%H%M%S}.gpickle")

    # ── mask ──
    @staticmethod
    def _mask_change(before: np.ndarray, after: np.ndarray):
        ch = np.asarray(before, bool) != np.asarray(after, bool)
        if not ch.any():
            return None
        rr, cc = np.nonzero(ch); r0, r1, c0, c1 = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        b, a = np.asarray(before, bool)[r0:r1, c0:c1], np.asarray(after, bool)[r0:r1, c0:c1]
        return dict(bbox_rc=[r0, c0, r1, c1], shape=list(b.shape), before=np.packbits(b), after=np.packbits(a),
                    n_added=int((a & ~b).sum()), n_removed=int((b & ~a).sum()))

    def _write_mask(self, m: np.ndarray) -> None:
        import tifffile
        fd, tmp = tempfile.mkstemp(prefix=".patch_mask_", suffix=".tif", dir=str(self.mask_path.parent)); os.close(fd)
        tifffile.imwrite(tmp, (np.asarray(m, bool) * 255).astype(np.uint8), compression="zlib"); os.replace(tmp, self.mask_path)

    def _apply_mask(self, mc: dict, which: str) -> None:
        import tifffile
        m = tifffile.imread(self.mask_path) > 127; r0, c0, r1, c1 = mc["bbox_rc"]; n = (r1 - r0) * (c1 - c0)
        m[r0:r1, c0:c1] = np.unpackbits(mc[which])[:n].reshape(mc["shape"]).astype(bool)
        self._write_mask(m)

    # ── accept / revert ──
    def accept(self, G_old: nx.Graph, G_new: nx.Graph, *, kind: str, note: str = "", polygon=None, report: Optional[dict] = None,
               mask_before: Optional[np.ndarray] = None, mask_after: Optional[np.ndarray] = None, dump=None) -> dict:
        """Record G_old → G_new as the next patch, write G_new to the graph file (atomic) and, if given, the mask.
        ``G_old`` must be the graph currently on disk (the caller's loaded copy).  ``dump(fh)`` overrides pickling
        (the viewer strips in-memory attributes).  Returns the log entry."""
        d = diff(G_old, G_new)
        if is_empty(d):
            raise ValueError("no change to record")
        self._ensure_base()
        pid, stamp = self._next_id(), datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mc = None
        if mask_before is not None and mask_after is not None:
            if self.mask_path is None:
                raise ValueError("mask change given but PatchLog has no mask_path")
            mc = self._mask_change(mask_before, mask_after)
        with gzip.open(self.dir / f"{pid}.pkl.gz", "wb") as fh:
            pickle.dump(dict(diff=d, mask=mc, report=report or {}), fh, protocol=4)
        if mc is not None:
            from ..mask.journal import record
            self._write_mask(mask_after); record(self.mask_path, mask_before, mask_after, note=f"patch {pid} ({kind}): {note}", stamp=stamp)
        self._write_graph(G_new, dump)
        entry = dict(id=pid, kind=kind, note=note, stamp=stamp, status="applied", graph=self.graph_path.name,
                     polygon=None if polygon is None else np.asarray(polygon, float).round(2).tolist(), counts=summary(d),
                     mask=None if mc is None else dict(bbox_rc=mc["bbox_rc"], n_added=mc["n_added"], n_removed=mc["n_removed"]),
                     report={k: v for k, v in (report or {}).items() if isinstance(v, (int, float, str, bool, type(None)))})
        self._write_log(self.entries() + [entry])
        return entry

    def conflicts(self, pid: str) -> list:
        """IDs of later applied patches that touch the same nodes / edges as ``pid``."""
        es = self.entries(); i = [e["id"] for e in es].index(pid)
        n0, e0 = touched(self.load_diff(pid)["diff"]); out = []
        for e in es[i + 1:]:
            if e["status"] != "applied":
                continue
            n1, e1 = touched(self.load_diff(e["id"])["diff"])
            if (n0 & n1) or (e0 & e1):
                out.append(e["id"])
        return out

    def revert(self, G: nx.Graph, pid: str, dump=None) -> nx.Graph:
        """Revert patch ``pid`` on G (the graph currently on disk); writes the graph (and mask) and marks the entry."""
        es = self.entries(); ent = next(e for e in es if e["id"] == pid)
        if ent["status"] != "applied":
            raise ValueError(f"patch {pid} is not applied")
        c = self.conflicts(pid)
        if c:
            raise ValueError(f"patch {pid} cannot be reverted: later patch(es) {c} build on it — revert those first")
        P = self.load_diff(pid); G2 = G.copy(); revert(G2, P["diff"])
        if P.get("mask") is not None:
            self._apply_mask(P["mask"], "before")
            from ..mask.journal import record
            import tifffile
            # journal the restore as its own entry (the journal is append-only)
            m_after = tifffile.imread(self.mask_path) > 127
            mc = P["mask"]; r0, c0, r1, c1 = mc["bbox_rc"]; n = (r1 - r0) * (c1 - c0)
            m_before = m_after.copy(); m_before[r0:r1, c0:c1] = np.unpackbits(mc["after"])[:n].reshape(mc["shape"]).astype(bool)
            record(self.mask_path, m_before, m_after, note=f"revert patch {pid}")
        self._write_graph(G2, dump)
        ent["status"] = "reverted"; ent["reverted"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._write_log(es)
        return G2
