"""Carry flow measurements from a previously analysed graph onto an edited graph.

After graph editing (+ condense), every vessel whose centreline is unchanged keeps its per-tile measurements;
edited or new vessels arrive unmeasured, so ``plexusflow analyze --skip-analyzed`` measures only those.
Labels made in the viewer live on the analysed graph and are carried too: node ``boundary_type`` (matched by node
position, ≤ ``tol_px``) and edge ``radial_uncertain`` (through the same vessel matching).  A label already present on
the edited graph wins.

Matching is geometric, so it survives node renumbering and edge-orientation changes:
  1. exact match of the centreline rounded to 0.1 px (either direction);
  2. otherwise same endpoints (≤ ``tol_px``), same number of points and every point within ``tol_px``.
``flow_from`` / ``flow_to`` are remapped to the edited graph's node IDs through the matched endpoints.
"""
from __future__ import annotations
from ..compat import sidecar_inputs, LEGACY_SIDECAR_KEY
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
    by_key_all = {}                                    # every new vessel by centreline (for tag carry-over)
    for u, v, d in G_new.edges(data=True):
        if d.get("path") is not None:
            p_ = np.asarray(d["path"], float); by_key_all[_key(p_)] = (u, v); by_key_all[_key(p_[::-1])] = (u, v)
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
    # viewer tags on vessels: radial_uncertain (old vessels without measurements are matched exactly too)
    n_tag = 0
    for u, v, d in G_old.edges(data=True):
        if d.get("radial_uncertain") and d.get("path") is not None:
            if (u, v) not in edge_map:
                hit = by_key_all.get(_key(np.asarray(d["path"], float)))
                if hit is not None:
                    edge_map[(u, v)] = hit
            if (u, v) in edge_map and not G_new.edges[edge_map[(u, v)]].get("radial_uncertain"):
                G_new.edges[edge_map[(u, v)]]["radial_uncertain"] = True; n_tag += 1
    # viewer labels on nodes: boundary_type (source / sink), matched by position
    n_lab = 0
    lab_old = [(n, d["boundary_type"]) for n, d in G_old.nodes(data=True) if d.get("boundary_type")]
    if lab_old:
        from scipy.spatial import cKDTree
        new_ids = list(G_new.nodes); tree = cKDTree(np.array([[G_new.nodes[n]["x"], G_new.nodes[n]["y"]] for n in new_ids], float))
        for n, bt in lab_old:
            dist, j = tree.query([G_old.nodes[n]["x"], G_old.nodes[n]["y"]])
            if dist <= tol_px and not G_new.nodes[new_ids[j]].get("boundary_type"):
                G_new.nodes[new_ids[j]]["boundary_type"] = bt; n_lab += 1
    for k in ("tile_f0_piv", "measurement_batch"):
        if k in G_old.graph:
            G_new.graph[k] = copy.deepcopy(G_old.graph[k])
    stats = dict(edges_new_graph=G_new.number_of_edges(), carried_exact=n_exact, carried_tolerance=n_tol,
                 to_measure=n_new, measurements_carried=n_meas,
                 old_edges_with_measurements=len(ends), old_edges_not_found=len(ends) - sum(1 for e in edge_map if G_old.edges[e].get("measurements_piv")),
                 boundary_labels_carried=n_lab, boundary_labels_old=len(lab_old), uncertain_tags_carried=n_tag,
                 uncertain_tags_old=sum(1 for *_, d in G_old.edges(data=True) if d.get("radial_uncertain")))
    return stats, edge_map


def remap_sidecars(sidecar_dir, edge_map):
    """Rewrite profile-fit sidecars so their keys use the edited graph's edge IDs; drop vessels that no longer exist."""
    import gzip, pickle
    from pathlib import Path
    n = 0
    for f in sorted(Path(sidecar_dir).glob("tile_*.pkl.gz")):
        with gzip.open(f, "rb") as fh:
            S = pickle.load(fh)
        inp = sidecar_inputs(S); S.pop(LEGACY_SIDECAR_KEY, None)
        S["profile_inputs"] = {edge_map[e]: v for e, v in inp.items() if e in edge_map}
        with gzip.open(f, "wb") as fh:
            pickle.dump(S, fh, protocol=4)
        n += 1
    return n
