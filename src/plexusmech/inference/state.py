"""PipelineState — single source of truth for the graph the inference /
simulation pipeline reads, plus a small caching layer keyed on the
current ``radial_uncertain`` tagging.

Ported from ``plexusflow/scripts/inference/pipeline_state.py`` on
2026-09-30 as part of the migration into ``plexusflow.inference`` (now ``plexusmech.inference``, 2026-10 split)
package. The bookkeeping is unchanged:

  * Bucket 1 — independent of tagging (normalization scalars, tile
    gauges, H1/H2 sync, boundary-node bc_harmonics, etc.). These live
    on the graph directly; we just read them.
  * Bucket 2 — derived from tagging (porous nodes, outlets, templates,
    α_j fit, carved sim edge flows, ``*_sim_carved`` fields). Cached
    keyed by the tagging fingerprint under ``cache/``.
  * Bucket 3 — legacy HYBRID-only fields (``_h_*_sim_intact``, etc.).
    Fully regenerable; dropped on rebuild.

Usage:
    from plexusmech.inference import PipelineState
    ps = PipelineState.load()                    # default: viewer graph
    print(ps.summary())
    U, outlets, outlet_comp, A_a, V_a, comps = ps.outlets()   # cached
    Gc, srcs, snks = ps.carved_graph()           # cached

Deferred: ``outlets()``, ``reduced_network()``, ``measured_dc_targets()``,
and ``carved_graph()`` still call the legacy scripts
``fit_dc_porous_sources.py`` and ``run_sim_carved_outlets.py`` via a
``sys.path`` bootstrap. When those two scripts are themselves ported
into this package (``plexusflow.inference.alpha_fit`` and
``plexusflow.inference.porous`` respectively), this shim goes away.
"""
from __future__ import annotations
import sys, pickle, hashlib
from pathlib import Path
from typing import Optional, Any
import numpy as np

_ROOT = Path("/Users/amayer/Library/CloudStorage/GoogleDrive-amayer"
             "@flatironinstitute.org/My Drive/Python/Yolk Sac Code")
_STAGE = Path("/Users/amayer/Library/CloudStorage/GoogleDrive-amayer"
              "@flatironinstitute.org/My Drive/PlexusFlow21S")

# Default input for the inference pipeline. Points at the most recent
# tile_normalize output that has both the normalized fields AND the
# current tagging state. See migration TODO: eventually resolve
# canonical graph flavor per stage.
DEFAULT_VIEWER_GRAPH = _STAGE / "graph" / "graph_vessels_piv_normalized.h2flipfix.gpickle"

# The frozen HYBRID snapshot (READ-ONLY). Kept as a fallback so the
# published paper baseline can still be reproduced bit-for-bit.
FROZEN_HYBRID_GRAPH = (_ROOT / "plexusflow/scripts/inference/data"
                       / "graph_V_templates_HYBRID.gpickle")

# Cache lives beside the frozen data dir but is its own subtree. Nothing
# in here is a "publishable" artifact — everything is regenerable from
# the tagging fingerprint.
CACHE_ROOT = _ROOT / "plexusflow/scripts/inference/cache"


def _ensure_scripts_importable() -> None:
    """Bootstrap: ensure the legacy inference scripts are importable so
    the deferred functions (``fit_dc_porous_sources``,
    ``run_sim_carved_outlets``) can be reached. Removes need for the
    caller to fiddle with ``sys.path``."""
    for sub in ("plexusflow/scripts", "plexusflow/scripts/inference",
                "PerTileFlow"):
        p = _ROOT / sub
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def tagging_fingerprint(G) -> tuple[str, int]:
    """Deterministic sha256 (first 16 hex chars) of the sorted list of
    tagged undirected edges, plus the tag count. A fresh tagging
    fingerprint means a fresh cache entry — nothing else."""
    tagged = sorted(tuple(sorted((int(u), int(v))))
                    for u, v, d in G.edges(data=True)
                    if d.get('radial_uncertain'))
    h = hashlib.sha256()
    for uv in tagged:
        h.update(f"{uv[0]},{uv[1]};".encode())
    return h.hexdigest()[:16], len(tagged)


def _cache_dir_for(fingerprint: str) -> Path:
    d = CACHE_ROOT / fingerprint
    d.mkdir(parents=True, exist_ok=True)
    return d


class PipelineState:
    """Lightweight container for the loaded graph + its tagging-derived
    bookkeeping. Everything below the ``# --- derived (bucket 2) ---``
    line is regenerable; nothing above the line is written by this class.
    """

    graph_path: Path
    G: Any
    fingerprint: str
    n_tags: int

    def __init__(self, graph_path: Path, G):
        self.graph_path = Path(graph_path)
        self.G = G
        self.fingerprint, self.n_tags = tagging_fingerprint(G)
        self._cache_dir = _cache_dir_for(self.fingerprint)
        # --- derived (bucket 2) — lazy ---
        self._outlets_bundle = None            # (U, outlets, outlet_comp, A_a, V_a, comps)
        self._reduced = None                    # (Nodes, Redges, idx, N, V_in, Ls, edge_meta, A_mat_dc)
        self._Qmeas_dc = None                   # (Q_meas_dc, w_cv)
        self._carved = None                     # (Gc, srcs, snks)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, graph_path: Optional[Path] = None) -> "PipelineState":
        p = Path(graph_path) if graph_path is not None else DEFAULT_VIEWER_GRAPH
        if not p.exists():
            raise FileNotFoundError(p)
        G = pickle.loads(p.read_bytes())
        return cls(p, G)

    # ------------------------------------------------------------------
    # cache access
    # ------------------------------------------------------------------
    def cache_dir(self) -> Path:
        return self._cache_dir

    def cache_path(self, name: str) -> Path:
        return self._cache_dir / name

    def cache_save(self, name: str, obj) -> Path:
        p = self.cache_path(name)
        p.write_bytes(pickle.dumps(obj))
        return p

    def cache_load(self, name: str, default=None):
        p = self.cache_path(name)
        if not p.exists():
            return default
        return pickle.loads(p.read_bytes())

    # ------------------------------------------------------------------
    # bucket-2 derivations (cached in-memory; use cache_save/load for disk)
    # ------------------------------------------------------------------
    def outlets(self):
        """Run ``fit_dc._find_porous_and_outlets`` once and memoize the
        result on this state object."""
        if self._outlets_bundle is not None:
            return self._outlets_bundle
        _ensure_scripts_importable()
        import fit_dc_porous_sources as fit_dc
        self._outlets_bundle = fit_dc._find_porous_and_outlets(self.G)
        return self._outlets_bundle

    def reduced_network(self):
        """Build the reduced-network Laplacian, edge_meta, and DC Green's
        matrix. Memoized — depends on the tagging alone."""
        if self._reduced is not None:
            return self._reduced
        _ensure_scripts_importable()
        import fit_dc_porous_sources as fit_dc
        U, outlets, _, A_a, V_a, _ = self.outlets()
        Nodes, Redges, idx, N_, V_in = fit_dc._build_reduced_network(
            self.G, U, outlets, set(A_a), set(V_a))
        Ls, edge_meta = fit_dc._assemble_laplacian(Redges, idx, N_)
        A_mat_dc = fit_dc._greens_functions(Ls, edge_meta, outlets, idx, N_)
        self._reduced = (Nodes, Redges, idx, N_, V_in, Ls, edge_meta, A_mat_dc)
        return self._reduced

    def measured_dc_targets(self, weight_scheme: str = 'inv_cv2'):
        """The (Q_meas_dc, w_cv) target for the DC fit. Depends on the
        reduced network (→ tagging) and on the fixed tile-normalization
        scalars (bucket 1, unchanging). Memoized per weight_scheme."""
        key = ('Q_meas_dc', weight_scheme)
        if self._Qmeas_dc is not None and self._Qmeas_dc[0] == key:
            return self._Qmeas_dc[1]
        _ensure_scripts_importable()
        import fit_dc_porous_sources as fit_dc
        _, _, _, _, _, _, edge_meta, _ = self.reduced_network()
        Q_meas_dc, w_cv = fit_dc._assemble_target(
            edge_meta, self.G,
            self.G.graph.get('tile_f0_piv', {}),
            self.G.graph.get('tile_gauges', {}),
            weight_scheme=weight_scheme)
        self._Qmeas_dc = (key, (Q_meas_dc, w_cv))
        return Q_meas_dc, w_cv

    def carved_graph(self):
        """Build the carved graph (drop tagged edges + A anchors, promote
        outlets to sources). Memoized."""
        if self._carved is not None:
            return self._carved
        _ensure_scripts_importable()
        from run_sim_carved_outlets import _build_carved_graph
        U, outlets, _, A_a, _, _ = self.outlets()
        Gc = _build_carved_graph(self.G, U, outlets, sorted(A_a))
        srcs = [n for n, d in Gc.nodes(data=True)
                if d.get('boundary_type') == 'source']
        snks = [n for n, d in Gc.nodes(data=True)
                if d.get('boundary_type') == 'sink']
        self._carved = (Gc, srcs, snks)
        return self._carved

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def summary(self) -> str:
        U, outlets, outlet_comp, A_a, V_a, comps = self.outlets()
        lines = [
            f"[state] graph = {self.graph_path}",
            f"[state] nodes = {self.G.number_of_nodes()}  edges = {self.G.number_of_edges()}",
            f"[state] tagging fingerprint = {self.fingerprint}  ({self.n_tags} tagged edges)",
            f"[state] porous nodes |U| = {len(U)}   components = {len(comps)}  "
            f"sizes = {[len(c) for c in comps]}",
            f"[state] outlets ({len(outlets)}): {outlets}",
            f"[state] A anchors {sorted(A_a)}   V anchors {sorted(V_a)}",
            f"[state] cache dir = {self._cache_dir}",
        ]
        # note if this fingerprint is the FROZEN HYBRID one
        if self.fingerprint == '565ce66a00103f5d':
            lines.append("[state] NOTE: this tagging matches the FROZEN HYBRID snapshot "
                          "(2026-09-23 lock baseline).")
        return "\n".join(lines)
