"""
KCL-based boundary reconstruction.

When the anatomical boundary arm sits in a low-SNR OF region, direct
measurement of the boundary Q(t) is noisy or wrong.  But by
conservation, the flow entering the boundary leaf equals the sum of
flows entering the SUBGRAPH containing the leaf via edges that DO have
clean measurements.

This module auto-expands a subgraph rooted at a boundary leaf until
every "interface" edge (subgraph → outside) has SNR above a threshold,
then reconstructs the boundary's Q(t) and harmonic phasors from those
interface edges.

Public API
----------
- ``find_kcl_subgraph(G, leaf, snr_threshold, max_depth)`` — grow
  subgraph until interface edges are all high-SNR.  Returns
  ``(subgraph_nodes, interface_edges)``.
- ``reconstruct_boundary_via_kcl(G, leaf, ...)`` — call
  ``find_kcl_subgraph`` and sum interface Q(t) / harmonics.  Returns a
  dict with the reconstructed fields (bc_harmonics, Q_t, mean_Q, etc.).
- ``apply_kcl_reconstruction(G, leaf, ...)`` — call
  ``reconstruct_boundary_via_kcl`` and write the result to
  ``G.nodes[leaf]['bc_harmonics']`` + provenance fields.

Provenance fields written to the leaf node:
- ``bc_harmonics``: complex ndarray[n_harm+1] = [DC, H1, H2, H3, ...]
- ``bc_source``: 'kcl_reconstruction'
- ``bc_source_edges``: list of interface (plex_node, sub_node) tuples
- ``bc_source_subgraph``: list of subgraph node ids
- ``bc_source_snr_min``: minimum SNR across interface edges (composite
  SNR of the reconstruction, roughly)

Provenance on the arm edge:
- ``Q_t_kcl_reconstructed``: reconstructed time series
- ``mean_Q_kcl_reconstructed``: DC of the reconstruction

Sign convention
---------------
Stored ``bc_harmonics`` uses the **solver-ready** convention:
positive DC = flow INJECTED INTO the network at that boundary.
- Source (arterial) leaves: positive DC (blood enters network).
- Sink (venous) leaves: negative DC (blood leaves network).

Internally we sum interface Q(t) with signs "positive INTO subgraph"
(= positive INTO leaf, opposite of solver convention), then negate
the sum at storage time so what lands in ``bc_harmonics`` is
solver-ready.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from ..io.measurement_view import MeasurementView


DEFAULT_SNR_THRESHOLD_DB = 15.0
DEFAULT_MAX_DEPTH = 5
DEFAULT_N_HARMONICS = 3


def _meas_snr_db(m) -> float:
    """Return the DC-harmonic-fit SNR (dB) for a measurement dict, via
    MeasurementView.  Returns -inf if unavailable.  Under v4 this is
    ``view.snr.dc_db``; under v3 this maps back to ``snr_dc_fit_db``
    (the closest analog of the legacy ``snr_harm_fit_db`` sort key)."""
    snr = MeasurementView(m).snr.dc_db
    if snr is None or not np.isfinite(snr):
        return -np.inf
    return float(snr)


def _best_meas(G, u, v, min_snr: Optional[float] = None):
    """Return the highest-SNR measurement on edge (u, v), or None if
    the edge has no measurement (or if ``min_snr`` is given and no
    measurement passes).

    Sort key is now ``view.snr.dc_db`` (v4 nested harmonic-fit DC SNR;
    was ``snr_harm_fit_db`` under v3)."""
    ed = G.edges[u, v]
    ml = ed.get('measurements_piv') or []
    ml_valid = []
    for m in ml:
        v_m = MeasurementView(m)
        mQ = v_m.harm.mean_Q
        snr = v_m.snr.dc_db
        if (mQ is not None and np.isfinite(mQ)
                and snr is not None and np.isfinite(snr)):
            ml_valid.append(m)
    if not ml_valid:
        return None
    best = max(ml_valid, key=_meas_snr_db)
    if min_snr is not None and _meas_snr_db(best) < min_snr:
        return None
    return best


def _edge_is_measurable(G, u, v, snr_threshold: float) -> bool:
    """True iff the edge has a measurement with SNR ≥ threshold."""
    return _best_meas(G, u, v, min_snr=snr_threshold) is not None


def find_kcl_subgraph(
    G: nx.Graph,
    leaf,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Tuple[Set, List[Tuple]]:
    """Grow a subgraph rooted at ``leaf`` by absorbing any node reached
    through an edge below ``snr_threshold`` (or unmeasurable).

    Stops when every edge crossing the subgraph boundary is measurable
    at or above the threshold, OR when ``max_depth`` BFS hops have been
    taken.

    Returns
    -------
    (subgraph_nodes, interface_edges)
        ``subgraph_nodes`` is the set of node ids inside the subgraph
        (includes ``leaf``).  ``interface_edges`` is a list of
        ``(outside_node, subgraph_node)`` tuples — the outside node is
        the plexus-side endpoint of each edge that crosses the
        boundary.  Sign convention: sum of Q flowing FROM the outside
        node TOWARD the subgraph node equals net Q entering the
        subgraph = Q entering the leaf.
    """
    subgraph: Set = {leaf}
    frontier = deque([leaf])
    depth = 0
    while frontier and depth < max_depth:
        next_frontier: deque = deque()
        for node in list(frontier):
            for nb in G.neighbors(node):
                if nb in subgraph:
                    continue
                if not _edge_is_measurable(G, node, nb, snr_threshold):
                    # Absorb the far end into the subgraph.
                    subgraph.add(nb)
                    next_frontier.append(nb)
        frontier = next_frontier
        depth += 1
    interface_edges: List[Tuple] = []
    for node in subgraph:
        for nb in G.neighbors(node):
            if nb in subgraph:
                continue
            # nb is outside → this edge crosses the boundary.
            interface_edges.append((nb, node))
    return subgraph, interface_edges


def reconstruct_boundary_via_kcl(
    G: nx.Graph,
    leaf,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
    max_depth: int = DEFAULT_MAX_DEPTH,
    n_harmonics: int = DEFAULT_N_HARMONICS,
) -> Optional[dict]:
    """Sum interface Q(t) + harmonic phasors, sign-normalized so
    positive = flowing INTO the subgraph (= INTO the leaf).

    Returns None if no valid interface edges were found.

    Returned dict keys:
      - ``bc_harmonics``: complex ndarray of length ``n_harmonics + 1``
        (DC + H1..Hn), sign-normalized so DC positive = into leaf.
      - ``Q_t``: reconstructed time series (only if all interface
        edges share the same length; None otherwise).
      - ``mean_Q``: DC of the reconstruction (from mean_Q sum).
      - ``subgraph``: list of subgraph node ids.
      - ``interface_edges``: list of (outside_node, subgraph_node) tuples.
      - ``snr_min``, ``snr_med`` (dB): SNR stats over interface edges.
      - ``n_interface``: number of interface edges used.
    """
    subgraph, interface = find_kcl_subgraph(G, leaf, snr_threshold, max_depth)
    if not interface:
        return None

    Q_t_stack: List[np.ndarray] = []
    DC_sum = 0.0
    harm_sum = np.zeros(n_harmonics, dtype=complex)
    snrs: List[float] = []
    tiles_seen: List[int] = []
    f0s: List[float] = []
    n_used = 0
    for plex, sub in interface:
        m = _best_meas(G, plex, sub, min_snr=snr_threshold)
        if m is None:
            continue
        view = MeasurementView(m)
        sign = 1.0 if view.flow_to == sub else -1.0
        DC_sum += sign * float(view.harm.mean_Q)
        harm = view.harm.harmonics or []
        for k in range(min(n_harmonics, len(harm))):
            amp = float(harm[k].get('amp', 0.0))
            phi = float(harm[k].get('phi', 0.0))
            harm_sum[k] += sign * amp * np.exp(1j * phi)
        Q_t = view.Q_t
        if Q_t is not None:
            Q_t_stack.append(sign * np.asarray(Q_t, dtype=float))
        snrs.append(_meas_snr_db(m))
        if view.tile_id is not None:
            tiles_seen.append(view.tile_id)
        f0_v = view.harm.f0_hz
        if f0_v is not None:
            f0s.append(float(f0_v))
        n_used += 1

    if n_used == 0:
        return None

    # Sums are "into subgraph" (= into leaf).  Solver-ready
    # convention is "injected into network", which is the opposite.
    # Negate at storage time so bc_harmonics is directly usable by
    # the solver.
    bc_harmonics = -np.concatenate([
        np.array([complex(DC_sum, 0.0)]),
        harm_sum,
    ])

    # Sum Q_t only if all interfaces share length + tiles are same
    # (identical timebase); otherwise leave Q_t as None.
    Q_t_reconstructed: Optional[np.ndarray] = None
    if Q_t_stack:
        lens = {len(q) for q in Q_t_stack}
        if len(lens) == 1 and len(set(tiles_seen)) == 1:
            Q_t_reconstructed = np.sum(Q_t_stack, axis=0)

    return dict(
        bc_harmonics = bc_harmonics,
        Q_t = Q_t_reconstructed,
        mean_Q = float(DC_sum),
        subgraph = sorted(subgraph),
        interface_edges = list(interface),
        snr_min = float(min(snrs)),
        snr_med = float(np.median(snrs)),
        n_interface = n_used,
        tiles_seen = sorted(set(tiles_seen)),
        f0s = f0s,
    )


def apply_kcl_reconstruction(
    G: nx.Graph,
    leaf,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
    max_depth: int = DEFAULT_MAX_DEPTH,
    n_harmonics: int = DEFAULT_N_HARMONICS,
    replace_direct: bool = True,
) -> Optional[dict]:
    """Apply KCL reconstruction to ``leaf`` and write the results to
    the graph.  Returns the reconstruction dict (same shape as
    :func:`reconstruct_boundary_via_kcl`), or None if no reconstruction
    was possible.

    Parameters
    ----------
    replace_direct : bool, default True
        If True, overwrite ``G.nodes[leaf]['bc_harmonics']`` with the
        reconstruction.  If False, only add provenance fields but leave
        the direct measurement in place.

    Fields written to the leaf node (whether or not ``replace_direct``):
      - ``bc_harmonics_kcl``: the reconstruction (complex ndarray)
      - ``bc_source_edges``: interface edge list
      - ``bc_source_subgraph``: subgraph node list
      - ``bc_source_snr_min`` / ``_snr_med``: SNR stats
      - ``bc_source_n_interface``: interface edge count

    If ``replace_direct=True`` also sets:
      - ``bc_harmonics``: overwritten with reconstruction
      - ``bc_source``: 'kcl_reconstruction'

    Fields written to the arm edge (leaf → first neighbour):
      - ``Q_t_kcl_reconstructed`` (if the reconstruction produced a
        time series)
      - ``mean_Q_kcl_reconstructed``
    """
    result = reconstruct_boundary_via_kcl(
        G, leaf,
        snr_threshold=snr_threshold,
        max_depth=max_depth,
        n_harmonics=n_harmonics,
    )
    if result is None:
        return None

    G.nodes[leaf]["bc_harmonics_kcl"] = result["bc_harmonics"]
    G.nodes[leaf]["bc_source_edges"] = result["interface_edges"]
    G.nodes[leaf]["bc_source_subgraph"] = result["subgraph"]
    G.nodes[leaf]["bc_source_snr_min"] = result["snr_min"]
    G.nodes[leaf]["bc_source_snr_med"] = result["snr_med"]
    G.nodes[leaf]["bc_source_n_interface"] = result["n_interface"]

    if replace_direct:
        G.nodes[leaf]["bc_harmonics"] = result["bc_harmonics"]
        G.nodes[leaf]["bc_source"] = "kcl_reconstruction"

    # Write time-series alias on the arm edge if available.
    if G.degree(leaf) == 1 and result["Q_t"] is not None:
        nb = next(iter(G.neighbors(leaf)))
        arm = G.edges[leaf, nb]
        arm["Q_t_kcl_reconstructed"] = result["Q_t"]
        arm["mean_Q_kcl_reconstructed"] = result["mean_Q"]

    return result


def apply_kcl_where_snr_low(
    G: nx.Graph,
    direct_snr_threshold_db: float = 15.0,
    kcl_snr_threshold: float = 15.0,
    max_depth: int = DEFAULT_MAX_DEPTH,
    n_harmonics: int = DEFAULT_N_HARMONICS,
    boundary_types: Iterable[str] = ("source", "sink"),
) -> List:
    """Iterate every boundary leaf; if its direct arm SNR is below
    ``direct_snr_threshold_db``, apply KCL reconstruction.

    Returns the list of leaves that were reconstructed.
    """
    reconstructed: List = []
    for n, d in G.nodes(data=True):
        if G.degree(n) != 1:
            continue
        if d.get("boundary_type") not in set(boundary_types):
            continue
        nb = next(iter(G.neighbors(n)))
        direct = _best_meas(G, n, nb)
        direct_snr = _meas_snr_db(direct) if direct is not None else -np.inf
        if direct_snr < direct_snr_threshold_db:
            result = apply_kcl_reconstruction(
                G, n,
                snr_threshold=kcl_snr_threshold,
                max_depth=max_depth,
                n_harmonics=n_harmonics,
            )
            if result is not None:
                reconstructed.append(n)
    return reconstructed


# ============================================================================
# Recursive junction-KCL — used by the editing viewer's mark-boundary mode.
# ============================================================================

def _sum_flows_out_of_node(G, node, other_neighbors, n_harmonics):
    """Sum signed 'flow OUT of node' over the specified neighbors.

    Returns (DC_sum, harm_sum, Q_t_sum, snrs, edges_used).
    Sign convention: +mean_Q if measurement has flow_from=node,
                     -mean_Q if flow_from=neighbor.
    So DC_sum > 0 means net flow leaves `node`.
    """
    DC_sum = 0.0
    harm_sum = np.zeros(n_harmonics, dtype=complex)
    snrs: List[float] = []
    edges_used: List[Tuple] = []
    Q_t_sum = None
    for nbr in other_neighbors:
        m = _best_meas(G, node, nbr)
        if m is None:
            continue
        view = MeasurementView(m)
        sign = 1.0 if view.flow_from == node else -1.0
        dc = view.harm.mean_Q
        if dc is not None and np.isfinite(dc):
            DC_sum += sign * float(dc)
        for k, h in enumerate((view.harm.harmonics or [])[:n_harmonics]):
            amp = float(h.get('amp', 0.0) or 0.0)
            phi = float(h.get('phi', 0.0) or 0.0)
            harm_sum[k] += sign * amp * np.exp(1j * phi)
        snrs.append(_meas_snr_db(m))
        edges_used.append((node, nbr))
        Q_t = view.Q_t
        if Q_t is not None:
            q = sign * np.asarray(Q_t, dtype=float)
            if Q_t_sum is None:
                Q_t_sum = q.copy()
            elif len(Q_t_sum) == len(q):
                Q_t_sum = Q_t_sum + q
    return DC_sum, harm_sum, Q_t_sum, snrs, edges_used


def compute_bc_harmonics_recursive(
    G,
    node,
    boundary_type: str,
    n_harmonics: int = 3,
    max_depth: int = 5,
    snr_threshold_db: float = 3.0,
    _visited=None,
    _depth: int = 0,
) -> Optional[dict]:
    """Recursive junction-KCL: compute bc_harmonics for a boundary node,
    walking further into the network when incident branches are
    unmeasured or below `snr_threshold_db`.

    At each level:
      - If `node` is degree-1 leaf: walk arm to first junction; recurse
        into the junction's other incident edges.
      - If `node` is degree ≥ 2 (interior boundary or intermediate
        junction during recursion): iterate over all incident edges.
        For each edge that has a good direct measurement, contribute
        its (signed) flow to the KCL sum.  For each edge that's
        unmeasured or below SNR threshold, RECURSE — walk to the other
        end and characterize that branch's flow via KCL at that node.

    Recursion is depth-limited (`max_depth`) and cycle-safe (`_visited`).

    Returns a dict:
      { 'bc_harmonics': complex ndarray shape (n_harmonics+1,),  # solver-ready
        'bc_Q_t': ndarray or None,
        'bc_snr_db': float,          # weakest-link SNR across contributing edges
        'bc_source': 'recursive_kcl',
        'edges_used': list of (u, v) tuples,
        'max_depth_reached': int,
        'n_edges': int,
        'n_recursions': int,          # how many nested recursions were invoked
      }
    or None if no contributing edges could be found.

    Sign convention: bc_harmonics in solver-ready convention (positive =
    injected into network).  See :func:`set_bc_harmonics_direct` in
    `scripts/canonicalize_to_s21_schema.py` for the reference derivation.
    """
    # Inlined from pertile.mosaic.scaffold.find_first_junction — walk from
    # a degree-1 anatomical leaf through degree-2 nodes until hitting a
    # degree ≥ 3 junction.  Returns (junction_id, [arm_edges]) or (None, ...)
    # if the walk exits at another leaf / cycles / exceeds max_walk=100.
    def find_first_junction(_G, _leaf, _max_walk=100):
        if _leaf not in _G or _G.degree(_leaf) != 1:
            return None, []
        _arm = []
        _prev = _leaf
        _cur = next(iter(_G.neighbors(_leaf)))
        for _ in range(_max_walk):
            _arm.append((_prev, _cur))
            _deg = _G.degree(_cur)
            if _deg == 1:
                return None, _arm
            if _deg >= 3:
                return _cur, _arm
            _nb = [_n for _n in _G.neighbors(_cur) if _n != _prev]
            if not _nb:
                return None, _arm
            _prev, _cur = _cur, _nb[0]
        return None, _arm

    if _visited is None:
        _visited = set()
    if node in _visited or _depth > max_depth:
        return None
    _visited = _visited | {node}

    deg = G.degree(node)

    if deg == 1 and _depth == 0:
        # Degree-1 leaf at the boundary: walk arm to first junction, then
        # process the junction as the effective "measurement site".
        junction, arm_edges = find_first_junction(G, node)
        if junction is None:
            # Degenerate arm — fall back to the direct arm's one measurement.
            nbr = next(iter(G.neighbors(node)))
            DC_sum, harm_sum, Q_t_sum, snrs, edges_used = _sum_flows_out_of_node(
                G, node, [nbr], n_harmonics)
            if not snrs:
                return None
            return dict(
                bc_harmonics=np.concatenate(
                    [[complex(DC_sum, 0.0)], harm_sum]),
                bc_Q_t=Q_t_sum,
                bc_snr_db=float(min(snrs)),
                bc_source='direct_arm_fallback',
                edges_used=edges_used,
                max_depth_reached=0,
                n_edges=len(snrs),
                n_recursions=0,
            )
        # Move to junction, exclude the arm-back neighbor.
        arm_side_neighbor = arm_edges[-1][0]
        _visited = _visited | {arm_side_neighbor}
        return _recursive_junction_sum(
            G, junction, exclude_neighbor=arm_side_neighbor,
            boundary_type=boundary_type,
            n_harmonics=n_harmonics, max_depth=max_depth,
            snr_threshold_db=snr_threshold_db,
            _visited=_visited, _depth=_depth + 1,
        )

    # Interior boundary (deg ≥ 2) OR intermediate junction during recursion:
    # sum all incident edges (or all except the exclude).
    return _recursive_junction_sum(
        G, node, exclude_neighbor=None,
        boundary_type=boundary_type,
        n_harmonics=n_harmonics, max_depth=max_depth,
        snr_threshold_db=snr_threshold_db,
        _visited=_visited, _depth=_depth,
    )


def _recursive_junction_sum(
    G, junction, exclude_neighbor, boundary_type,
    n_harmonics, max_depth, snr_threshold_db,
    _visited, _depth,
):
    """Recursive body: sum "flow out of junction" over all incident edges
    (except the one going back to the boundary), recursing into
    unmeasured branches."""
    DC_sum = 0.0
    harm_sum = np.zeros(n_harmonics, dtype=complex)
    snrs: List[float] = []
    edges_used: List[Tuple] = []
    Q_t_sum = None
    max_depth_reached = _depth
    n_recursions = 0

    other_neighbors = [n for n in G.neighbors(junction)
                       if n != exclude_neighbor and n not in _visited]
    for nbr in other_neighbors:
        m = _best_meas(G, junction, nbr)
        m_snr = _meas_snr_db(m) if m is not None else -np.inf
        if m is not None and m_snr >= snr_threshold_db:
            # Direct measurement is usable — add to KCL sum.
            view = MeasurementView(m)
            sign = 1.0 if view.flow_from == junction else -1.0
            dc = view.harm.mean_Q
            if dc is not None and np.isfinite(dc):
                DC_sum += sign * float(dc)
            for k, h in enumerate((view.harm.harmonics or [])[:n_harmonics]):
                amp = float(h.get('amp', 0.0) or 0.0)
                phi = float(h.get('phi', 0.0) or 0.0)
                harm_sum[k] += sign * amp * np.exp(1j * phi)
            snrs.append(m_snr)
            edges_used.append((junction, nbr))
            Q_t = view.Q_t
            if Q_t is not None:
                q = sign * np.asarray(Q_t, dtype=float)
                if Q_t_sum is None:
                    Q_t_sum = q.copy()
                elif len(Q_t_sum) == len(q):
                    Q_t_sum = Q_t_sum + q
        else:
            # Unmeasured or bad SNR — recurse to characterize this branch.
            if _depth >= max_depth:
                continue
            sub = compute_bc_harmonics_recursive(
                G, nbr, boundary_type,
                n_harmonics=n_harmonics,
                max_depth=max_depth,
                snr_threshold_db=snr_threshold_db,
                _visited=_visited | {junction},
                _depth=_depth + 1,
            )
            n_recursions += 1
            if sub is None:
                continue
            # sub['bc_harmonics'] is in solver-ready convention for `nbr`
            # (positive = injected into network AT nbr).
            #
            # Sign derivation (previously wrong; 2026-07-22 fix):
            #   sub_bc(nbr) = net flow injected into network at nbr
            #               = net flow entering nbr from EXTERNAL side
            #               (= junction J side, since nbr's other neighbors
            #                are the "network" during recursion).
            #   By interior KCL at nbr:
            #       (flow into nbr from J) + (flow into nbr from network) = 0
            #   sub_bc = flow J→nbr = "flow out of junction toward nbr".
            #   So:  Q_out(J → nbr) = +sub_bc  (NOT negated).
            #
            # The previous `-sub_bc` inverted the sign of every recursed
            # branch — verify against a manual KCL at J for a boundary
            # node before touching this again.
            sub_bc = sub['bc_harmonics']
            DC_sum += sub_bc[0].real
            for k in range(min(n_harmonics, len(sub_bc) - 1)):
                harm_sum[k] += sub_bc[k + 1]
            snrs.append(sub['bc_snr_db'])
            edges_used.append((junction, nbr))
            edges_used.extend(sub.get('edges_used', []))
            max_depth_reached = max(max_depth_reached, sub.get('max_depth_reached', 0))
            n_recursions += sub.get('n_recursions', 0)
            sub_Q_t = sub.get('bc_Q_t')
            if sub_Q_t is not None:
                q = np.asarray(sub_Q_t, dtype=float)
                if Q_t_sum is None:
                    Q_t_sum = q.copy()
                elif len(Q_t_sum) == len(q):
                    Q_t_sum = Q_t_sum + q

    if not snrs:
        return None

    # Convert "flow out of junction" DC_sum → solver-ready bc_harmonics
    # for the ORIGINAL boundary node.  Recall: bc = +DC_sum (source
    # positive, sink negative) per set_bc_harmonics_direct's derivation
    # when DC_sum is "flow out of junction toward the boundary arm".
    # Here DC_sum is "flow out of junction toward OTHER branches" (or
    # into them via recursion); by KCL at junction, flow toward arm =
    # -DC_sum.  Then bc = flow toward arm = -DC_sum? Wait: for a sink
    # at leaf B, flow physically goes junction→arm→B→outside, so
    # "flow out of junction toward arm" > 0 → bc = -("flow out of j
    # toward arm") = negative ✓ ...
    #
    # Actually the direct derivation matches junction-KCL branch in
    # canonicalize_to_s21_schema.py: bc = +DC_sum where DC_sum is
    # "flow out of junction summed over OTHER branches".  Which is
    # what we computed here.  Keep sign as-is.
    bc_harmonics = np.concatenate([[complex(DC_sum, 0.0)], harm_sum])
    return dict(
        bc_harmonics=bc_harmonics,
        bc_Q_t=Q_t_sum,
        bc_snr_db=float(min(snrs)),
        bc_source='recursive_kcl' if n_recursions else 'junction_kcl',
        edges_used=edges_used,
        max_depth_reached=max_depth_reached,
        n_edges=len(snrs),
        n_recursions=n_recursions,
    )
