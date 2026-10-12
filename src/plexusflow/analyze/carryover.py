"""Carry flow measurements from a previously analysed graph onto an edited graph.

After graph editing (+ condense), every vessel whose centreline is unchanged keeps its per-tile measurements;
edited or new vessels arrive unmeasured, so ``plexusflow analyze --skip-analyzed`` measures only those.
Node attributes (boundary labels, stubs …) come from the edited graph — labelling never needs re-analysis.

Matching is geometric, so it survives node renumbering and edge-orientation changes:
  1. exact match of the centreline rounded to 0.1 px (either direction);
  2. otherwise same endpoints (≤ ``tol_px``), same number of points and every point within ``tol_px``.
``flow_from`` / ``flow_to`` are remapped to the edited graph's node IDs through the matched endpoints.
"""
from __future__ import annotations
import copy
import numpy as np


def _key(path):
    return tuple(map(tuple, np.round(np.asarray(path, float), 1)))


def carry_measurements(G_new, G_old, tol_px: float = 1.0):
    """Copy ``measurements_piv`` from G_old to matching edges of G_new (in place).

    Returns (stats dict, edge_map {old (u, v): new (u, v)}).
    """
    by_key, ends = {}, []
    for u, v, d in G_old.edges(data=True):
        if not d.get("measurements_piv") or d.get("path") is None:
            continue
        p = np.asarray(d["path"], float)
        by_key[_key(p)] = (u, v, False); by_key[_key(p[::-1])] = (u, v, True)
        ends.append((u, v, p))
    end_arr = np.array([np.r_[p[0], p[-1]] for _, _, p in ends]) if ends else np.zeros((0, 4))
    edge_map, n_exact, n_tol, n_new, n_meas = {}, 0, 0, 0, 0
    for u, v, d in G_new.edges(data=True):
        if d.get("path") is None:
            n_new += 1; continue
        p = np.asarray(d["path"], float); hit = by_key.get(_key(p))
        if hit is None and len(end_arr):
            for flip, q in ((False, p), (True, p[::-1])):
                dd = np.hypot(*(end_arr[:, :2] - q[0]).T) + np.hypot(*(end_arr[:, 2:] - q[-1]).T)
                for i in np.flatnonzero(dd <= 2 * tol_px):
                    ou, ov, op = ends[i]
                    if len(op) == len(q) and np.max(np.hypot(*(op - q).T)) <= tol_px:
                        hit = (ou, ov, flip); break
                if hit is not None:
                    break
            if hit is not None:
                n_tol += 1
        elif hit is not None:
            n_exact += 1
        if hit is None:
            n_new += 1; continue
        ou, ov, _ = hit
        # map old endpoint IDs → new endpoint IDs by position
        def xy(G, n):
            return np.array([G.nodes[n]["x"], G.nodes[n]["y"]], float)
        nmap = ({ou: u, ov: v} if np.hypot(*(xy(G_old, ou) - xy(G_new, u))) + np.hypot(*(xy(G_old, ov) - xy(G_new, v)))
                <= np.hypot(*(xy(G_old, ou) - xy(G_new, v))) + np.hypot(*(xy(G_old, ov) - xy(G_new, u))) else {ou: v, ov: u})
        ms = copy.deepcopy(G_old.edges[ou, ov]["measurements_piv"])
        for m in ms:
            m["flow_from"] = nmap.get(m.get("flow_from"), m.get("flow_from"))
            m["flow_to"] = nmap.get(m.get("flow_to"), m.get("flow_to"))
        d["measurements_piv"] = ms; n_meas += len(ms); edge_map[(ou, ov)] = (u, v)
    for k in ("tile_f0_piv", "measurement_batch"):
        if k in G_old.graph:
            G_new.graph[k] = copy.deepcopy(G_old.graph[k])
    stats = dict(edges_new_graph=G_new.number_of_edges(), carried_exact=n_exact, carried_tolerance=n_tol,
                 to_measure=n_new, measurements_carried=n_meas,
                 old_edges_with_measurements=len(ends), old_edges_not_found=len(ends) - len(edge_map))
    return stats, edge_map


def remap_sidecars(sidecar_dir, edge_map):
    """Rewrite Kloosterman sidecars so their keys use the edited graph's edge IDs; drop vessels that no longer exist."""
    import gzip, pickle
    from pathlib import Path
    n = 0
    for f in sorted(Path(sidecar_dir).glob("tile_*.pkl.gz")):
        with gzip.open(f, "rb") as fh:
            S = pickle.load(fh)
        S["kl_inputs"] = {edge_map[e]: v for e, v in S.get("kl_inputs", {}).items() if e in edge_map}
        with gzip.open(f, "wb") as fh:
            pickle.dump(S, fh, protocol=4)
        n += 1
    return n
