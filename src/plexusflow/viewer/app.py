"""Read-only mosaic viewer — exploration only, no editing or compute.

Designed for two audiences:

  1. **Intern onboarding** — they can see the mosaic, color edges by any
     measured field, click an edge to inspect its Q(t) and metadata.  No
     buttons that change the graph; no compute that takes minutes.

  2. **Publication / data sharing** — a single self-contained Qt+napari
     app a reviewer can launch on a gpickle to explore the underlying
     measurements without learning the analysis pipeline.

Run from CLI:

    python -m pertile.viewer.mosaic_readonly_app <graph.gpickle>         [--tiff <stitched.tif>] [--initial-field mean_Q]

Or from Python:

    from plexusflow.viewer.app import ReadOnlyMosaicViewer
    v = ReadOnlyMosaicViewer(graph_path, tiff_path=None,
                              initial_field='mean_Q')
    v.run()
"""
from __future__ import annotations

from ._common import (  # noqa: F401 - re-exported: app.py was the single viewer module
    _note_suppressed,
    CATEGORICAL_FIELDS,
    DEFAULT_FIELDS,
    Dict,
    FLOW_MISMATCH_COLOR,
    HARMONIC_CACHE_VERSION,
    HARMONIC_CLASS_COLORS,
    HARMONIC_KEYS,
    HARMONIC_LABELS,
    HARMONIC_SNR_THRESHOLD,
    List,
    MeasurementView,
    NO_DATA_COLOR,
    Optional,
    PROPERTY_DEFS,
    Path,
    QUALITY_RANK,
    RADIAL_UNCERTAIN_COLOR,
    SIM_FIELD_MAP,
    SIM_INTERNAL_KEYS,
    SIM_PRESSURE_FIELDS,
    TIER_COLORS,
    TILE_RAW_HEIGHT,
    TILE_RAW_WIDTH,
    Tuple,
    _TiedVShim,
    _actual_boundary_Q_dc,
    _best_measurement,
    _combo_valid,
    _harmonic_class,
    _harmonic_class_label,
    _harmonic_fit_full,
    _harmonic_snrs,
    _measurement_usable,
    _osi_from_harmonics,
    _pertile_candidates,
    _pick_existing_dir,
    _place_figure_top_right,
    _print_dc_bc_diagnostic,
    _resolve_edge_field,
    _safe_float,
    _video_glob,
    argparse,
    json,
    math,
    mcolors,
    np,
    pickle,
    plt,
    sys,
)
from .simulate import SimulateMixin
from .inference import InferenceMixin
from .edit import EditMixin
from .optical_flow import OpticalFlowMixin
from .timeseries import TimeseriesMixin


class ReadOnlyMosaicViewer(SimulateMixin, InferenceMixin, EditMixin, OpticalFlowMixin, TimeseriesMixin):
    """napari-based mosaic viewer with field colouring + edge inspection.

    Two capability levels (``mode``):
      'view'      — purely for viewing: never writes any file, no fresh optical flow, no tagging or boundary
                    editing, no simulations.  The default, and the only mode allowed on a frozen release.
      'workbench' — adds fresh OF on a selected vessel, radial-uncertainty tagging, boundary marking (Edit tab)
                    and saving those to the stage graph; the Simulate tab appears only if `plexusmech` is installed.
    """


    def __init__(
        self,
        graph_path: Path,
        tiff_path: Optional[Path] = None,
        tile_positions: Optional[Path] = None,
        video_dir: Optional[Path] = None,
        video_pattern: Optional[str] = None,
        of_video_dir: Optional[Path] = None,
        of_video_pattern: Optional[str] = None,
        initial_field: str = 'mean_Q',
        cache_harmonic_class: bool = True,
        force_recompute_harmonic_class: bool = False,
        rotate_cw: bool = False,
        rotate_ccw: bool = False,
        mode: str = 'view',
    ):
        self.graph_path = Path(graph_path)
        if mode not in ('view', 'workbench'):
            raise ValueError(f"mode must be 'view' or 'workbench', not {mode!r}")
        # A frozen release (MANIFEST.json in the stage folder) is never written: workbench is refused there.
        self.is_release = (self.graph_path.parent.parent / 'MANIFEST.json').exists()
        if mode == 'workbench' and self.is_release:
            print("  ⚠ This is a frozen data release — opening read-only (workbench edits need a stage folder).")
            mode = 'view'
        self.mode = mode
        self.workbench = mode == 'workbench'
        import importlib.util as _ilu
        self.mech_available = _ilu.find_spec('plexusmech') is not None
        if not self.workbench:
            cache_harmonic_class = False      # view mode never writes the graph
        self.tiff_path = Path(tiff_path) if tiff_path else None
        self.tile_positions_path = Path(tile_positions) if tile_positions else None
        # video_dir / of_video_dir may each be a list of CANDIDATE locations
        # (or a comma-separated string); use whichever exists on this machine,
        # so one command works on desktop and laptop.
        self.video_dir, self._video_candidates = _pick_existing_dir(video_dir)
        self.video_pattern = video_pattern
        if self._video_candidates and self.video_dir is None:
            print("  ⚠ Video overlay: none of the candidate dirs exist on this "
                  "machine: " + ", ".join(str(c) for c in self._video_candidates))
        elif self.video_dir is not None and len(self._video_candidates) > 1:
            print(f"  Video dir: {self.video_dir}  "
                  f"(first existing of {len(self._video_candidates)} candidates)")
        if of_video_dir is not None:
            self.of_video_dir, _ = _pick_existing_dir(of_video_dir)
        else:
            self.of_video_dir = self.video_dir
        self.of_video_pattern = of_video_pattern or self.video_pattern
        self.cache_harmonic_class = bool(cache_harmonic_class)
        self.force_recompute_harmonic_class = bool(force_recompute_harmonic_class)
        self.rotate_cw = bool(rotate_cw)
        self.rotate_ccw = bool(rotate_ccw)
        if self.rotate_cw and self.rotate_ccw:
            raise ValueError("Pass at most one of --rotate-cw / --rotate-ccw.")
        self.G = self._load_graph()
        # Reconciliation outputs (DC Q', AC field) from the stage/release results folder; the
        # 'Reconciled' data source reads them.  In-memory only (`_h_*_rec` edge attrs).
        self._rec = self._load_reconciliation()
        self.mosaic = self._load_tiff() if self.tiff_path else None
        self.tiles = self._load_tile_positions() if self.tile_positions_path else {}
        self.edge_list: List[Tuple[int, int]] = list(self.G.edges())
        # Currently-overlaid video tile (napari layer + tile_id) — only
        # one at a time; clicking a new edge replaces the previous overlay.
        self._video_layer = None
        self._video_tile_id: Optional[int] = None
        # Precompute midpoints for nearest-edge lookup.
        self._edge_midpoints = np.array([
            [0.5 * (self.G.nodes[u]['x'] + self.G.nodes[v]['x']),
             0.5 * (self.G.nodes[u]['y'] + self.G.nodes[v]['y'])]
            for (u, v) in self.edge_list
        ])
        # Precompute per-edge `harmonic_class` (0/1/2/3 = number of
        # statistically resolved cardiac harmonics).  Stored on the
        # in-memory graph only — never written back to the gpickle.
        # Cheap enough to do once at startup (~30 s for ~6k edges).
        self._precompute_harmonic_classes()
        # Precompute measured viscous dissipation per edge from PIV
        # Q_t + geometry.  Stored as `_meas_dissipation` (W) on edge
        # attrs, in-memory only.
        self._precompute_measured_dissipation()
        # ── Four-selector field model ──────────────────────────────
        # Source already added below as `self.current_source`.
        # Quantity: 'Q' or 'P'.  P only valid in Simulated source.
        # Property: see PROPERTY_DEFS.  Some are harmonic-keyed.
        # Harmonic: 'DC' / 'H1' / 'H2' / 'H3'.  Ignored for non-keyed
        # properties.
        self.current_quantity: str = 'Q'
        self.current_property: str = 'magnitude'
        self.current_harmonic: str = 'DC'
        # Backwards-compat: `current_field` is now a synthesized string
        # so legacy checks like `'phase' in field` and
        # `field in SIM_PRESSURE_FIELDS` keep working.  See the
        # `current_field` @property below.  `self.fields` is no longer
        # needed (the selectors are static) but is kept as an empty
        # list to avoid breaking anything that iterates it.
        self.fields: List[str] = []
        # initial_field may still set the starting Property if it
        # matches a known key.
        if (initial_field
                and any(p[0] == initial_field for p in PROPERTY_DEFS)):
            self.current_property = initial_field
        self.log_scale = False
        self.show_nodes = False
        self.show_kcl = False
        self._kcl_layer = None
        self._edges_layer = None
        self._nodes_layer = None
        self._click_marker_layer = None
        self._tile_boundaries_layer = None
        self._tile_labels_layer = None
        self.show_tile_boundaries = False
        self.show_tile_labels = False
        # Edge value labels — show the current property value at each
        # edge's midpoint.  Hidden by default and ALSO requires a single
        # tile to be selected (Tile filter ≠ "All tiles") to render —
        # otherwise the labels overlap into clutter at mosaic scale.
        self.show_edge_labels = False
        self._edge_labels_layer = None
        # Network-wide reliability filtering is intentionally out of the
        # UI for now — the cached per-harmonic Z layers (`_h_Z_DC`,
        # `_h_Z_HN`) are the right basis for a future continuous,
        # threshold-at-display-time filter.  See discussion thread for
        # the design (four-layer Z field, no precomputed tiers).
        # edge (u,v) → {'amp_DC', 'Z_DC', 'amp_H1', 'phase_H1', 'Z_H1', ...}
        # Populated for the currently-loaded tile only (either by the
        # "Browse tiles" video-load path, or eagerly by the tile filter
        # combo).  Used both by the time-series inspect panel for
        # cross-edge-comparable phase data, and by the tile filter to
        # restrict measured-Q colormap fields to one tile's values.
        self._tile_harmonic_cache: Dict[Tuple[int, int], dict] = {}
        # Tile filter state — None ⇒ "All tiles", colour by best
        # measurement across tiles (default).  When set to an int,
        # measured-Q fields read per-tile values from
        # `_tile_harmonic_cache`; edges with no measurement on the
        # filtered tile render as grey.  Doesn't affect sim or geometry.
        self.current_tile_filter: Optional[int] = None
        # Percentile threshold on per-edge total_snr (Var(fit)/Var(resid)).
        # Hides the bottom X % of edges by network-wide `_h_total_snr`.
        # The percentile is always computed from the graph-wide cached
        # best-of-edge values — independent of whether a tile filter is
        # active.  0 = no filtering, default.
        # Absolute-dB threshold on Q_H1_snr_db.  Sentinel −20 dB = off.
        self.total_snr_pct_filter: float = -20.0
        # (Legacy cache, kept for compatibility; no longer read.)
        self._cached_total_snr_threshold: Optional[float] = None
        # Absolute-dB threshold on Q_H2_snr_db.  Sentinel −20 dB = off.
        self.total_snr_h2_pct_filter: float = -20.0
        self._cached_total_snr_h2_threshold: Optional[float] = None
        # Percentile threshold on per-edge viscous dissipation Φ (W).
        # Hides the bottom X % of edges by network-wide `_meas_dissipation`.
        # 0 = no filtering, default.  Cached threshold + stats are
        # invalidated when the filter value or graph changes.
        self.dissipation_pct_filter: float = 0.0
        self._cached_dissipation_threshold: Optional[float] = None
        self._cached_dissipation_stats: Optional[Dict[str, float]] = None
        # Edges queued for the side-by-side comparison plot.  Each
        # entry is (u, v); duplicates ignored.  Cleared via the
        # "Clear comparison" button or when the user explicitly resets.
        self._comparison_edges: List[Tuple[int, int]] = []
        self._comparison_marker_layer = None
        self._comparison_edge_overlay_layer = None
        # Boundary nodes — arterial sources ('source') and venous sinks
        # ('sink').  Identified from the `boundary_type` node attribute
        # set by the analysis pipeline; both have a complex
        # `bc_harmonics` array (DC + H1 + H2 + H3).
        self._source_nodes: List[int] = [
            n for n, d in self.G.nodes(data=True)
            if d.get('boundary_type') == 'source']
        self._sink_nodes: List[int] = [
            n for n, d in self.G.nodes(data=True)
            if d.get('boundary_type') == 'sink']
        self._boundary_nodes = self._source_nodes + self._sink_nodes
        self._source_layer = None
        self._sink_layer = None
        print(f"  boundary: {len(self._source_nodes)} sources (A), "
              f"{len(self._sink_nodes)} sinks (V)")
        # Simulation state (forward transmission-line solve).  Results
        # live IN-MEMORY only — never persisted to the gpickle.  Keys
        # use the `_sim_tmp_` prefix so they don't collide with the
        # editing-app's stored `_sim` fields.
        self._sim_active: bool = False
        self._sim_last_D: float = float('nan')
        self._sim_last_f0_hz: float = float('nan')
        # If the loaded graph carries persisted sim results (`_sim_tmp_*`
        # on any edge, or a `sim_meta` blob on G.graph), flip _sim_active
        # so the View tab's Simulated radio is enabled from the start.
        # No new solve is triggered — the fields are just already present.
        if self.G.graph.get('sim_meta'):
            self._sim_active = True
        else:
            for _u, _v, _d in self.G.edges(data=True):
                if '_sim_tmp_harmonics' in _d or '_sim_tmp_mean_Q' in _d:
                    self._sim_active = True
                    break
        if self._sim_active:
            _meta = self.G.graph.get('sim_meta') or {}
            self._sim_last_D = float(_meta.get('D_areal', float('nan')))
            self._sim_last_f0_hz = float(_meta.get('f0_hz', float('nan')))
        # Data source for the View tab field colormap + the comparison
        # plot.  'measured' uses PIV results from `measurements_piv`;
        # 'reconciled' uses the DC reconciliation + aligned AC field from
        # results/ (`_load_reconciliation`); 'sim' uses the in-memory
        # `_sim_tmp_*` outputs.  The Source radio in the View tab drives
        # this; 'reconciled' is disabled when no results are found, and
        # 'sim' is disabled until `_sim_active`.
        self.current_source: str = 'measured'
        # Mosaic height — used to invert node y so points align with the
        # image layer (mosaic uses display-y = row, graph stores math-y).
        # Falls back to max graph-y if no TIFF is loaded.
        if self.mosaic is not None:
            self.mosaic_height = int(self.mosaic.shape[0])
        else:
            ys = [self.G.nodes[n].get('y', 0) for n in self.G.nodes()]
            self.mosaic_height = int(max(ys) + 1) if ys else 0
        # Per-node "best" measured value for the currently-selected field;
        # populated by `_refresh_edges` (which already iterates edges) so
        # nodes can be colored by the mean of their incident edges.
        self._node_field_values: Dict[int, float] = {}


    # ── data loading ───────────────────────────────────────────────────
    def _meas_key(self, source: Optional[str] = None) -> str:
        """Per-edge measurements-list key (always `measurements_piv`; kept for callers)."""
        return 'measurements_piv'


    # ── Tile-filtered measured-Q resolver ─────────────────────────────
    def _resolve_tile_filtered_value(self, u, v, d: dict,
                                       prop: str) -> Optional[float]:
        """Read the per-tile equivalent of the requested measured-Q
        property from `_tile_harmonic_cache` (populated for whichever
        tile is currently selected as the filter).  Returns None when
        the edge has no measurement on that tile, when the property
        isn't well-defined per-tile (dissipation has no clean per-tile
        equivalent), or when the harmonic fit failed for that
        measurement.
        """
        # Geometry doesn't change per tile.
        if prop == 'radius':
            return _safe_float(d.get('radius'))
        if prop == 'length':
            return _safe_float(d.get('length'))
        # Frequency — pulled directly from the matching tile's
        # measurement (the per-tile harmonic cache reused this same
        # f0 during the fit, but doesn't store it back, so it's
        # cheaper to read the source).
        if prop == 'frequency':
            tile_id = self.current_tile_filter
            for mm in (d.get(self._meas_key()) or []):
                mv_mm = MeasurementView(mm)
                if mv_mm.tile_id == tile_id:
                    return _safe_float(mv_mm.harm.f0_hz)
            return None

        entry = self._tile_harmonic_cache.get((u, v))
        if entry is None:
            return None  # no measurement on the filtered tile for this edge

        if prop == 'PI':
            # Per-harmonic PI_k = 2·|Q_Hk| / |Q_DC|.  DC slot undefined.
            k_idx_pi = HARMONIC_KEYS.index(self.current_harmonic)
            if k_idx_pi == 0:
                return None
            a0 = abs(float(entry.get('amp_DC', 0.0) or 0.0))
            amp_k = float(entry.get(f'amp_H{k_idx_pi}', 0.0) or 0.0)
            if a0 < 1e-15:
                return None
            return 2.0 * amp_k / a0
        if prop == 'total_snr':
            return _safe_float(entry.get('total_snr'))
        if prop == 'snr_ac':
            tile_id = self.current_tile_filter
            for mm in (d.get(self._meas_key()) or []):
                mv_mm = MeasurementView(mm)
                if mv_mm.tile_id == tile_id:
                    return _safe_float(mv_mm.snr.h1_db)
            return None
        if prop == 'snr_total':
            tile_id = self.current_tile_filter
            for mm in (d.get(self._meas_key()) or []):
                mv_mm = MeasurementView(mm)
                if mv_mm.tile_id == tile_id:
                    return _safe_float(mv_mm.snr.dc_db)
            return None
        if prop == 'harmonic_class':
            snrs = {'DC': float(entry.get('Z_DC', 0.0) or 0.0),
                    'H1': float(entry.get('Z_H1', 0.0) or 0.0),
                    'H2': float(entry.get('Z_H2', 0.0) or 0.0),
                    'H3': float(entry.get('Z_H3', 0.0) or 0.0)}
            return _safe_float(_harmonic_class(snrs))
        if prop == 'dissipation':
            # `_meas_dissipation` is a graph-level aggregate from PIV
            # processing; no clean per-tile equivalent exists.  Return
            # None so the field reads as "not available at this tile".
            return None
        if prop == 'pulse_frac':
            # Per-tile: ½·Σ|H_k|² / (Q_DC² + ½·Σ|H_k|²) using THIS
            # tile's amplitudes from the harmonic cache.  R cancels,
            # so this is also the per-tile dissipation AC fraction.
            try:
                a0 = float(entry.get('amp_DC', 0.0) or 0.0)
                ac_sq = sum(float(entry.get(f'amp_H{k}', 0.0) or 0.0) ** 2
                             for k in (1, 2, 3))
                ac_e = 0.5 * ac_sq
                tot = a0 * a0 + ac_e
                return float(ac_e / tot) if tot > 0 else None
            except (TypeError, ValueError):
                return None
        if prop == 'dissipation_ac':
            # Per-tile: R_e · ½·Σ|H_k|² where R_e uses the per-tile
            # tile-corrected geometry (R_um_tile, L_um_tile) on the
            # tile's measurement entry.  Units: Pa·nL/s = nW.
            tile_id = self.current_tile_filter
            for mm in (d.get(self._meas_key()) or []):
                mv_mm = MeasurementView(mm)
                if mv_mm.tile_id != tile_id:
                    continue
                R_um = mv_mm.geom.R_um
                L_um = mv_mm.geom.L_um
                if R_um is None or L_um is None:
                    return None
                try:
                    R_um_f = float(R_um); L_um_f = float(L_um)
                except (TypeError, ValueError):
                    return None
                if not (np.isfinite(R_um_f) and np.isfinite(L_um_f)
                        and R_um_f > 0 and L_um_f > 0):
                    return None
                # G_e in nL/(Pa·s):  π R⁴ / (8 μ L)   with μ = 3.5e-3 Pa·s,
                # R, L in µm; convert µm⁴/µm = µm³ → 1e-9 nL·m³/m³ = 1e-9 nL/...
                # Actually: π R^4 (µm⁴) / (8μ L µm) = π·R⁴/(8μL) (µm³ / (Pa·s))
                # = 1e-9 nL/(Pa·s).  So G_nL_per_Pa_s = G_µm3_per_Pa_s × 1e-9.
                G_per_Pa_s_um3 = (np.pi * R_um_f ** 4 /
                                    (8.0 * 3.5e-3 * L_um_f))
                G_nL = G_per_Pa_s_um3 * 1e-9
                if G_nL <= 0: return None
                # Σ amp_H_k² from this tile's harmonics list.
                # v4 migration: was mm.get('harmonics')
                harms = mv_mm.harm.harmonics or []
                if harms:
                    sum_sq = sum(float(h.get('amp', 0.0) or 0.0) ** 2
                                  for h in harms[:3])
                else:
                    sum_sq = 0.0
                    for k in (1, 2, 3):
                        a = mm.get(f'Q_H{k}_amp')
                        if a is not None and np.isfinite(a):
                            sum_sq += float(a) ** 2
                if sum_sq <= 0:
                    return None
                phi_ac = 0.5 * sum_sq / G_nL    # nW
                return float(phi_ac)
            return None

        # Harmonic-keyed properties — keys in the per-tile cache match
        # the layout produced by `_harmonic_fit_full`.
        k_idx = HARMONIC_KEYS.index(self.current_harmonic)
        if prop == 'magnitude':
            if k_idx == 0:
                v_ = entry.get('amp_DC')
                return abs(_safe_float(v_)) if v_ is not None else None
            return _safe_float(entry.get(f'amp_H{k_idx}'))
        if prop == 'phase':
            if k_idx == 0:
                return None  # phase at DC undefined
            return _safe_float(entry.get(f'phase_H{k_idx}'))
        if prop == 'resolution':
            if k_idx == 0:
                return _safe_float(entry.get('Z_DC'))
            return _safe_float(entry.get(f'Z_H{k_idx}'))
        if prop == 'snr':
            # Per-harmonic SNR from THIS tile's measurement (canonical
            # fields Q_DC_snr_db / Q_Hk_snr_db on the per-tile entry).
            tile_id = self.current_tile_filter
            for mm in (d.get(self._meas_key()) or []):
                mv_mm = MeasurementView(mm)
                if mv_mm.tile_id == tile_id:
                    if k_idx == 0:
                        return _safe_float(mm.get('Q_DC_snr_db'))
                    return _safe_float(mm.get(f'Q_H{k_idx}_snr_db'))
            return None
        return None


    # ── AC-harmonic helpers (harmonic_fraction / waveform_av) ─────────
    def _edge_ac_harmonics(self, d, src, qty):
        """Complex AC harmonics [H1, H2, H3] for an edge under the given
        source/quantity, or None.  Measured-Q reads measurements_piv; sim-Q reads
        `_sim_tmp_harmonics`; sim-P reads `_sim_tmp_p_harmonics`
        (each stored as complex [DC, H1, H2, H3])."""
        if src == 'measured' and qty == 'Q':
            meas_key = 'measurements_piv'
            m = _best_measurement(d.get(meas_key))
            if m is None:
                return None
            # Canonical fields first; fall back to v2's harmonics list.
            # v4 migration: was m.get('harmonics')
            _harms = MeasurementView(m).harm.harmonics or []
            out = []
            for k in (1, 2, 3):
                a = m.get(f'Q_H{k}_amp')
                ph = m.get(f'Q_H{k}_phi')
                if (a is None or not np.isfinite(a)) and len(_harms) >= k:
                    a = _harms[k - 1].get('amp')
                    ph = _harms[k - 1].get('phi')
                if (a is None or ph is None
                        or not (np.isfinite(a) and np.isfinite(ph))):
                    return None
                out.append(float(a) * np.exp(1j * float(ph)))
            return np.array(out, dtype=complex)
        if src == 'sim' and qty == 'Q':
            h = d.get('_sim_tmp_harmonics')
            if h is None or len(h) < 4:
                return None
            return np.asarray(h[1:4], dtype=complex)
        if src == 'sim' and qty == 'P':
            p = d.get('_sim_tmp_p_harmonics')
            if p is None or len(p) < 4:
                return None
            return np.asarray(p[1:4], dtype=complex)
        return None


    def _boundary_edge_ac(self, nodes, src):
        """AC harmonics of the first boundary edge incident to one of
        `nodes` with a usable waveform — the 'DA segment' (source side) or
        the 'V boundary edge' (sink side)."""
        if not nodes:
            return None
        for n in nodes:
            if n not in self.G:
                continue
            for nbr in self.G.neighbors(n):
                ac = self._edge_ac_harmonics(self.G.edges[n, nbr], src, 'Q')
                if ac is not None and float(np.linalg.norm(ac)) > 1e-30:
                    return ac
        return None


    def _av_reference_harmonics(self, src):
        """(A_ref, V_ref) AC-harmonic reference vectors for the waveform_av
        field: A from an arterial (source) boundary edge, V from a venous
        (sink) boundary edge.  Returns None if either is unavailable.
        Cached per source; the cache is busted on sim re-run / clear."""
        key = (src,)
        cache = getattr(self, '_av_ref_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]
        A = self._boundary_edge_ac(getattr(self, '_source_nodes', None), src)
        V = self._boundary_edge_ac(getattr(self, '_sink_nodes', None), src)
        refs = (A, V) if (A is not None and V is not None) else None
        self._av_ref_cache = (key, refs)
        return refs


    # ── Field resolver (4-selector model) ─────────────────────────────
    def _resolve_field_value(self, u, v) -> Optional[float]:
        """Return the per-edge scalar for the current selector state,
        or None.  Centralised lookup used by both the colormap and the
        node-averaging pass.

        Routing matrix:
          aggregate properties  → top-level edge attr lookup
          harmonic-keyed (measured) → cached `_h_amp/phase/Z_H{k}` etc.
          harmonic-keyed (sim, Q)   → derived from `_sim_tmp_harmonics`
          harmonic-keyed (sim, P)   → derived from `_sim_tmp_p_harmonics`
        Invalid combos return None (rendered grey).
        """
        if not self.G.has_edge(u, v):
            return None
        if not _combo_valid(self.current_source, self.current_quantity,
                             self.current_property,
                             self.current_harmonic):
            return None
        d = self.G.edges[u, v]
        prop = self.current_property
        src = self.current_source
        qty = self.current_quantity

        # ── Carved-sim source: read `_h_*_sim_carved` fields directly.
        # These are populated by the Carved Sim tab's Run button.  Only
        # Q-flavour fields are supported (sim solves for edge fluxes).
        if src == 'sim_carved' and qty == 'Q':
            k_idx = HARMONIC_KEYS.index(self.current_harmonic)
            if prop == 'magnitude':
                key = ('_h_amp_DC_sim_carved' if k_idx == 0
                       else f'_h_amp_H{k_idx}_sim_carved')
                return _safe_float(d.get(key))
            if prop == 'phase':
                if k_idx == 0: return None
                return _safe_float(d.get(f'_h_phase_H{k_idx}_sim_carved'))
            if prop == 'PI':
                if k_idx == 0: return None
                amp = _safe_float(d.get(f'_h_amp_H{k_idx}_sim_carved'))
                dc  = _safe_float(d.get('_h_amp_DC_sim_carved'))
                if amp is None or dc is None or dc == 0: return None
                val = 2.0 * abs(amp) / abs(dc)
                return val if np.isfinite(val) else None
            # Fallthrough: unsupported prop for sim_carved → None
            return None

        # ── Reconciled source: DC Q' + tile-aligned AC field (`_h_*_rec`, see _load_reconciliation).
        # Phases only where the winner tile's time gauge is determined (bootstrap SD ≤ 30°).
        if src == 'reconciled':
            if qty != 'Q':
                return None
            k_idx = HARMONIC_KEYS.index(self.current_harmonic)
            if prop == 'rec_dc_sd':
                sd, dc = _safe_float(d.get('_rec_Qp_sd')), _safe_float(d.get('_h_amp_DC_rec'))
                return sd / abs(dc) if sd is not None and dc else None
            if prop == 'rec_gauge_sd':
                return _safe_float(d.get('_rec_gauge_sd'))
            if k_idx == 0:
                dc = _safe_float(d.get('_h_amp_DC_rec'))
                if dc is None:
                    return None
                return dc if prop == 'signed' else (abs(dc) if prop == 'magnitude' else None)
            amp = _safe_float(d.get(f'_h_amp_H{k_idx}_rec'))
            if prop == 'magnitude':
                return amp
            if prop == 'signed':
                ph = _safe_float(d.get(f'_h_phase_H{k_idx}_rec'))
                return amp * np.cos(ph) if amp is not None and ph is not None else None
            if prop == 'phase':
                return _safe_float(d.get(f'_h_phase_H{k_idx}_rec')) if d.get('_rec_phase_ok') else None
            if prop == 'PI':
                dc = _safe_float(d.get('_h_amp_DC_rec'))
                if amp is None or not dc:
                    return None
                return 2.0 * abs(amp) / abs(dc)
            return None

        # Tile-filter override: when a tile is selected, every measured-Q
        # value is taken from THAT tile's harmonic fit instead of the
        # best-across-tiles graph attrs.  Edges with no measurement on
        # the filtered tile fall through to None (rendered grey).
        if (src == 'measured' and qty == 'Q'
                and self.current_tile_filter is not None):
            return self._resolve_tile_filtered_value(u, v, d, prop)

        # BC-anchor phase override: for edges incident to a boundary
        # anchor node, use the tile-14-only measurement (raw φ +
        # tile-14 gauge) rather than the multi-tile average.  The BC
        # anchors live on tile 14, so its measurement is the trustworthy
        # one; the multi-tile average is polluted by low-SNR outside-
        # tile measurements that can be π-flipped, which then makes
        # sink stubs look out-of-sync with each other on the map.
        if (prop == 'phase' and src == 'measured'
                and qty == 'Q'):
            k_idx_bc = HARMONIC_KEYS.index(self.current_harmonic)
            if k_idx_bc >= 1:
                bt_u = self.G.nodes[u].get('boundary_type')
                bt_v = self.G.nodes[v].get('boundary_type')
                if bt_u in ('source', 'sink') or bt_v in ('source', 'sink'):
                    tg = self.G.graph.get('tile_gauges') or {}
                    g14 = tg.get(14) or {}
                    gk_re = f'z_H{k_idx_bc}_re'
                    gk_im = f'z_H{k_idx_bc}_im'
                    gauge_arg = 0.0
                    if (g14.get(gk_re) is not None
                            and g14.get(gk_im) is not None):
                        gauge_arg = float(np.arctan2(float(g14[gk_im]),
                                                     float(g14[gk_re])))
                    for m in (d.get('measurements_piv') or []):
                        if int(m.get('tile_id', -1)) != 14:
                            continue
                        for h in (m.get('harm', {}).get('harmonics') or []):
                            if int(h.get('k', 0)) != k_idx_bc:
                                continue
                            if h.get('phi') is not None:
                                return float(h['phi']) + gauge_arg
                        break

        # ── Harmonic fraction: |H_k|² / (|H1|²+|H2|²+|H3|²) ──
        if prop == 'harmonic_fraction':
            k_idx_hf = HARMONIC_KEYS.index(self.current_harmonic)
            if k_idx_hf == 0:
                return None  # DC is not part of the AC fraction
            ac = self._edge_ac_harmonics(d, src, qty)
            if ac is None:
                return None
            mags2 = np.abs(ac) ** 2
            tot = float(mags2.sum())
            if tot < 1e-30:
                return None
            return float(mags2[k_idx_hf - 1] / tot)

        # ── Waveform A↔V (aggregate; identical for every harmonic) ──
        if prop == 'waveform_av':
            if qty != 'Q':
                return None
            refs = self._av_reference_harmonics(src)
            if refs is None:
                return None
            e_ac = self._edge_ac_harmonics(d, src, qty)
            if e_ac is None:
                return None

            def _cossim(a, b):
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                if na < 1e-30 or nb < 1e-30:
                    return 0.0
                return float(np.real(np.vdot(b, a)) / (na * nb))

            return _cossim(e_ac, refs[0]) - _cossim(e_ac, refs[1])

        # ── Aggregate (non-harmonic-keyed) properties ──
        # 'mean' was dropped — use Magnitude @ DC for the |Q̄| / |P̄|
        # equivalent.
        if prop == 'PI':
            # Per-harmonic Gosling-style PI: PI_k = 2·|Q_Hk| / |Q_DC|.
            # k_idx = 0 (DC) is undefined; H1 is classic PI.
            k_idx_pi = HARMONIC_KEYS.index(self.current_harmonic)
            if k_idx_pi == 0:
                return None  # PI at DC is undefined
            if src == 'measured' and qty == 'Q':
                meas_key = 'measurements_piv'
                m = _best_measurement(d.get(meas_key))
                if m is None:
                    return None
                try:
                    # Compute from the harmonics list (falls back to the
                    # legacy top-level Q_H{k}_amp fields).
                    harms = (MeasurementView(m).harm.harmonics or [])
                    if k_idx_pi - 1 < len(harms):
                        amp_k = harms[k_idx_pi - 1].get('amp')
                    else:
                        amp_k = m.get(f'Q_H{k_idx_pi}_amp')
                    dc = (m.get('harm') or {}).get('mean_Q')
                    if dc is None:
                        dc = m.get('Q_DC')
                    if (amp_k is None or dc is None or float(dc) == 0):
                        return None
                    val = 2.0 * abs(float(amp_k)) / abs(float(dc))
                    return val if np.isfinite(val) else None
                except (TypeError, ValueError):
                    return None
            if src == 'sim' and qty == 'Q':
                # Sim path: derive PI_k from cached harmonics if available,
                # otherwise fall back to the cached H1-only PI.
                h = d.get('_sim_tmp_harmonics')
                if h is not None and k_idx_pi < len(h):
                    dc = abs(float(h[0].real))
                    if dc < 1e-30:
                        return None
                    return 2.0 * float(abs(h[k_idx_pi])) / dc
                if k_idx_pi == 1:
                    return _safe_float(d.get('_sim_tmp_PI'))
                return None
            if src == 'sim' and qty == 'P':
                p = d.get('_sim_tmp_p_harmonics')
                if p is None or k_idx_pi >= len(p):
                    return None
                p_dc = abs(float(p[0].real))
                if not np.isfinite(p_dc) or p_dc < 1e-12:
                    return None
                amp_k = float(abs(p[k_idx_pi]))
                if not np.isfinite(amp_k):
                    return None
                return 2.0 * amp_k / p_dc
            return None
        if prop == 'dissipation':
            if src == 'measured':
                return _safe_float(d.get('_meas_dissipation'))
            if src == 'sim':
                return _safe_float(d.get('_sim_tmp_dissipation'))
            return None
        if prop == 'dissipation_ac':
            if src == 'measured':
                # Persisted on v2 canonicals as dissipation_AC_piv (nW).
                return _safe_float(d.get('dissipation_AC_piv'))
            if src == 'sim':
                # Φ_AC = R_e · ½·Σ|H_k|² from sim Q harmonics.
                h = d.get('_sim_tmp_harmonics')
                if h is None or len(h) < 4:
                    return None
                sum_sq = float(sum(abs(complex(hk)) ** 2 for hk in h[1:4]))
                if sum_sq <= 0: return None
                # Get G_e from edge geometry (same as measured path uses).
                # v4 migration: was d.get('conductance_piv'); compute from
                # geometry via MeasurementView (πR⁴/(8μL), μ=3.5e-3 Pa·s).
                mv_ce = MeasurementView.best(d)
                if mv_ce is None:
                    return None
                R_um = mv_ce.geom.R_um; L_um = mv_ce.geom.L_um
                if R_um is None or L_um is None:
                    return None
                try:
                    G_e = math.pi * float(R_um) ** 4 / (8.0 * 3.5e-3 * float(L_um))
                except (TypeError, ValueError):
                    return None
                if not np.isfinite(G_e) or G_e <= 0:
                    return None
                return float(0.5 * sum_sq / float(G_e))
            return None
        if prop == 'pulse_frac':
            if src == 'measured':
                # Persisted on v2 canonicals as pulse_frac_piv ∈ [0, 1].
                # Same number whether interpreted as fraction of Q²-energy
                # or fraction of viscous dissipation — R_e cancels.
                return _safe_float(d.get('pulse_frac_piv'))
            if src == 'sim':
                # AC fraction from sim Q harmonics.
                h = d.get('_sim_tmp_harmonics')
                if h is None or len(h) < 4:
                    return None
                dc = abs(complex(h[0]))
                ac_sq = sum(abs(complex(hk)) ** 2 for hk in h[1:4])
                ac_e = 0.5 * ac_sq
                tot = dc * dc + ac_e
                return float(ac_e / tot) if tot > 0 else None
            return None
        if prop == 'OSI':
            # Reconstruct Q(t) from DC + harmonics over one period and
            # compute OSI_Q = ½·(1 − |∫Q|/∫|Q|).  Aggregate field — the
            # harmonic selector is ignored.
            if src == 'measured' and qty == 'Q':
                meas_key = 'measurements_piv'
                m = _best_measurement(d.get(meas_key))
                if m is None:
                    return None
                try:
                    dc_q = m.get('Q_DC')
                    if dc_q is None:
                        dc_q = (m.get('harm') or {}).get('mean_Q')
                    if dc_q is None or not np.isfinite(dc_q):
                        dc_q = 0.0
                    amps_phi = []
                    harms = (MeasurementView(m).harm.harmonics or [])
                    for k in (1, 2, 3):
                        a = m.get(f'Q_H{k}_amp')
                        p_ = m.get(f'Q_H{k}_phi')
                        if ((a is None or p_ is None) and len(harms) >= k):
                            a = harms[k - 1].get('amp')
                            p_ = harms[k - 1].get('phi')
                        if a is None or p_ is None: continue
                        if not (np.isfinite(a) and np.isfinite(p_)): continue
                        amps_phi.append((float(a), float(p_)))
                    if not amps_phi:
                        return None
                    return _osi_from_harmonics(float(dc_q), amps_phi)
                except (TypeError, ValueError):
                    return None
            if src == 'sim' and qty == 'Q':
                h = d.get('_sim_tmp_harmonics')
                if h is None or len(h) < 2:
                    return None
                try:
                    dc_q = float(np.real(h[0]))
                    amps_phi = []
                    for k in range(1, len(h)):
                        a = float(abs(h[k]))
                        p_ = float(np.angle(h[k]))
                        if np.isfinite(a) and np.isfinite(p_):
                            amps_phi.append((a, p_))
                    if not amps_phi:
                        return None
                    return _osi_from_harmonics(dc_q, amps_phi)
                except (TypeError, ValueError, AttributeError):
                    return None
            return None
        if prop == 'drop':
            return _safe_float(d.get('_sim_tmp_pressure_drop'))
        if prop == 'radius':
            # sim-mode override takes precedence (uniform-R / uniform-C).
            if src == 'sim':
                v_ = d.get('_sim_tmp_radius')
                if v_ is not None:
                    return _safe_float(v_)
            return _safe_float(d.get('radius'))
        if prop == 'length':
            if src == 'sim':
                v_ = d.get('_sim_tmp_length')
                if v_ is not None:
                    return _safe_float(v_)
            return _safe_float(d.get('length'))
        if prop == 'harmonic_class':
            return _safe_float(d.get('harmonic_class'))
        if prop == 'total_snr':
            # Cached during _precompute_harmonic_classes; only present
            # for measured Q edges that had a usable Q_t to fit.
            return _safe_float(d.get('_h_total_snr'))
        if prop == 'snr_ac':
            # AC-total SNR in dB: P(H1+H2+H3) / MS_BL.  Canonical edge attr.
            return _safe_float(d.get('snr_ac_fit_db'))
        if prop == 'snr_total':
            # DC+AC SNR in dB: P(DC+H1+H2+H3) / MS_BL.  Canonical edge attr.
            return _safe_float(d.get('snr_harm_fit_db'))
        if prop == 'frequency':
            if src == 'sim':
                return _safe_float(d.get('_sim_tmp_f0_hz'))
            # Measured: top-level edge attr is set during analysis from
            # the best measurement; fall back to scanning the per-tile
            # measurement list if it isn't there.
            v_ = d.get('f0_hz')
            if v_ is None:
                m = _best_measurement(d.get('measurements_piv'))
                if m is not None:
                    v_ = MeasurementView(m).harm.f0_hz
            return _safe_float(v_)

        # ── Harmonic-keyed (magnitude / phase / resolution) ──
        k_idx = HARMONIC_KEYS.index(self.current_harmonic)
        cache_suffix = ''
        if prop == 'magnitude':
            if src == 'measured' and qty == 'Q':
                if k_idx == 0:
                    v_ = d.get(f'_h_amp_DC{cache_suffix}')
                    if v_ is None:
                        meas_key = 'measurements_piv'
                        m = _best_measurement(d.get(meas_key))
                        if m is None:
                            return None
                        dc = m.get('Q_DC')
                        if dc is None:
                            dc = MeasurementView(m).harm.mean_Q
                        if dc is None:
                            dc = float('nan')
                        try:
                            return abs(float(dc))
                        except (TypeError, ValueError):
                            return None
                    try:
                        return abs(float(v_))
                    except (TypeError, ValueError):
                        return None
                return _safe_float(d.get(f'_h_amp_H{k_idx}{cache_suffix}'))
            if src == 'sim' and qty == 'Q':
                h = d.get('_sim_tmp_harmonics')
                if h is None or k_idx >= len(h):
                    return None
                if k_idx == 0:
                    return abs(float(h[0].real))
                return float(abs(h[k_idx]))
            if src == 'sim' and qty == 'P':
                h = d.get('_sim_tmp_p_harmonics')
                if h is None or k_idx >= len(h):
                    return None
                if k_idx == 0:
                    return abs(float(h[0].real))
                return float(abs(h[k_idx]))
            return None
        if prop == 'signed':
            # Signed real value (no |·|) — sim P/Q.  DC: the signed gauge
            # (can be < 0).  AC: in-phase component (sign is phase-ref-dependent).
            if src == 'sim' and qty == 'P':
                h = d.get('_sim_tmp_p_harmonics')
            elif src == 'sim' and qty == 'Q':
                h = d.get('_sim_tmp_harmonics')
            else:
                return None
            if h is None or k_idx >= len(h):
                return None
            return float(np.real(h[k_idx]))
        if prop == 'phase':
            if k_idx == 0:
                return None  # phase at DC undefined
            if src == 'measured' and qty == 'Q':
                return _safe_float(d.get(f'_h_phase_H{k_idx}{cache_suffix}'))
            if src == 'sim' and qty == 'Q':
                h = d.get('_sim_tmp_harmonics')
                if h is None or k_idx >= len(h):
                    return None
                return float(np.angle(h[k_idx]))
            if src == 'sim' and qty == 'P':
                h = d.get('_sim_tmp_p_harmonics')
                if h is None or k_idx >= len(h):
                    return None
                return float(np.angle(h[k_idx]))
            return None
        if prop == 'resolution':
            if src == 'measured' and qty == 'Q':
                if k_idx == 0:
                    return _safe_float(d.get(f'_h_Z_DC{cache_suffix}'))
                return _safe_float(d.get(f'_h_Z_H{k_idx}{cache_suffix}'))
            return None
        if prop == 'snr':
            # Per-harmonic SNR — canonical (top-level) edge attrs.
            if src == 'measured' and qty == 'Q':
                if k_idx == 0:
                    return _safe_float(d.get('Q_DC_snr_db'))
                return _safe_float(d.get(f'Q_H{k_idx}_snr_db'))
            return None
        return None


    # ── Synthesized "current_field" name ──────────────────────────────
    @property
    def current_field(self) -> str:
        """Synthesized canonical key from the 4 selectors.

        Format: ``<quantity>.<property>[@<harmonic>]`` (e.g.
        ``Q.magnitude@H2`` / ``P.mean`` / ``Q.PI``).  Legacy code paths
        check substrings like ``'phase' in field`` — these stay valid
        because the property key appears verbatim in the string.
        """
        parts = [self.current_quantity, self.current_property]
        prop = next((p for p in PROPERTY_DEFS
                      if p[0] == self.current_property), None)
        if prop is not None and prop[2]:
            return '.'.join(parts) + '@' + self.current_harmonic
        return '.'.join(parts)


    @current_field.setter
    def current_field(self, _val):
        """Setter kept as a no-op for legacy compatibility.  The actual
        selection is owned by the four selector state variables."""
        return


    def _current_field_label(self) -> str:
        """Human-readable label for the colorbar.

        Appends a SI unit suffix so colormap values are unambiguous
        when switching between Quantity (Q vs P) and Source (sim vs
        measured).  Properties whose unit is already in the display
        label (`SNR (dB)`, `Fundamental f₀ (Hz)`, `Radius (px)`,
        `Length (px)`) intentionally get no extra suffix to avoid
        duplication; categorical / dimensionless properties (PI, OSI,
        harmonic class) get nothing.
        """
        pdef = next((p for p in PROPERTY_DEFS
                      if p[0] == self.current_property), None)
        prop_lbl = pdef[1] if pdef else self.current_property
        qsym = self.current_quantity
        # (property, quantity) → unit string.  None / '' means "no unit
        # suffix needed".  Lookup order: (prop, qsym) → (prop, '*').
        UNITS = {
            ('magnitude', 'Q'):    'nL/s',
            ('magnitude', 'P'):    'Pa',
            ('signed',    'Q'):    'nL/s',
            ('signed',    'P'):    'Pa',
            ('phase',     '*'):    'rad',
            ('dissipation', '*'):  'W',
            ('drop',       '*'):   'Pa',
            # PI, OSI: dimensionless; SNR/SNR_AC/SNR_total: already (dB);
            # frequency: already (Hz); radius/length: already (px);
            # harmonic_class: categorical.  All omitted on purpose.
        }
        prop_key = self.current_property
        unit = UNITS.get((prop_key, qsym),
                          UNITS.get((prop_key, '*'), ''))
        # Compose source/harmonic suffix as before.
        if pdef is not None and pdef[2]:
            qual = f"  ({qsym}, {HARMONIC_LABELS[self.current_harmonic]})"
        else:
            qual = f"  ({qsym})"
        unit_str = f"  [{unit}]" if unit else ''
        return prop_lbl + qual + unit_str


    def _load_graph(self):
        print(f"Loading graph: {self.graph_path}")
        with open(self.graph_path, 'rb') as f:
            G = pickle.load(f)
        from ..paths import attach_external_traces      # release graphs keep Q(t) in export/traces.npz
        G = attach_external_traces(G, self.graph_path)
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G


    def _load_tiff(self):
        try:
            import tifffile
            print(f"Loading TIFF: {self.tiff_path}")
            img = tifffile.imread(self.tiff_path)
            print(f"  shape={img.shape}, dtype={img.dtype}")
            return img
        except Exception as e:
            print(f"  Could not load TIFF ({e}); proceeding without it.")
            return None


    def _load_tile_positions(self) -> Dict[int, dict]:
        """Parse tile_positions_manual.json → {tile_id: {translate_x,
        translate_y, scale_x, scale_y}}.  Also computes the global tile
        offset (min translate_x/y across all tiles) so video tiles can
        be placed in image-row coordinates: row = translate_y − offset_y.
        """
        try:
            with open(self.tile_positions_path) as f:
                tp = json.load(f)
        except Exception as e:
            print(f"  Could not load tile positions ({e}); video overlay disabled.")
            return {}
        tiles: Dict[int, dict] = {}
        for vid_str, entry in tp.get('tiles', {}).items():
            try:
                vid = int(vid_str)
            except ValueError:
                continue
            # Full 3x3 affine (last row [0,0,1]).  plexusflow's
            # positions.json stores this under 'affine_matrix' and tiles
            # can be rotated; the scale_x/y + translate_x/y fields alone
            # are just the diagonal of the affine and don't describe a
            # rotated tile's actual footprint on the mosaic.
            A = entry.get('affine_matrix')
            if A is not None:
                A = np.asarray(A, dtype=float)
            else:
                A = np.array([
                    [float(entry.get('scale_y', 1.0)), 0.0,
                     float(entry.get('translate_y', 0.0))],
                    [0.0, float(entry.get('scale_x', 1.0)),
                     float(entry.get('translate_x', 0.0))],
                    [0.0, 0.0, 1.0],
                ])
            tiles[vid] = {
                'translate_x': float(entry.get('translate_x', A[1, 2])),
                'translate_y': float(entry.get('translate_y', A[0, 2])),
                'scale_x':     float(entry.get('scale_x', A[1, 1])),
                'scale_y':     float(entry.get('scale_y', A[0, 0])),
                'affine':      A,
            }
        # Global tile offset — must match the fuse's `compute_world_bbox`
        # (plexusflow/fuse/feather.py): floor(min(corner_y)) - pad,
        # floor(min(corner_x)) - pad, using the four warped corners of
        # every tile (not just the translate).  `min(translate)` alone
        # misses the top rows of rotated tiles + the pad and shifts every
        # overlay by that amount.
        FUSE_PAD = 20
        if tiles:
            corners_yx = []
            for t in tiles.values():
                A = t.get('affine')
                if A is None:
                    continue
                raw = np.array([[0.0, 0.0],
                                [0.0, TILE_RAW_WIDTH],
                                [TILE_RAW_HEIGHT, TILE_RAW_WIDTH],
                                [TILE_RAW_HEIGHT, 0.0]])
                corners_yx.append(raw @ A[:2, :2].T + A[:2, 2])
            if corners_yx:
                corners_yx = np.concatenate(corners_yx, axis=0)
                self._tile_offset_y = float(np.floor(corners_yx[:, 0].min())
                                             - FUSE_PAD)
                self._tile_offset_x = float(np.floor(corners_yx[:, 1].min())
                                             - FUSE_PAD)
            else:
                self._tile_offset_x = min(t['translate_x']
                                          for t in tiles.values())
                self._tile_offset_y = min(t['translate_y']
                                          for t in tiles.values())
        else:
            self._tile_offset_x = 0.0
            self._tile_offset_y = 0.0
        print(f"  Loaded positions for {len(tiles)} tiles "
              f"from {self.tile_positions_path.name} "
              f"(offset={self._tile_offset_x:.0f}, {self._tile_offset_y:.0f})")
        return tiles


    def _get_video_path_for(self, tile_id: int) -> Optional[Path]:
        """Resolve the display video for ``tile_id``.

        Tries the primary ``video_dir`` first (typically the realtime
        downsampled bundle), then falls back to ``of_video_dir`` (the
        raw folder-of-frames) when the realtime file isn't present for
        this tile.  Lets you build the realtime bundle incrementally
        without leaving un-downsampled tiles without a display video.
        """
        if self.video_pattern is None:
            return None
        for base in (self.video_dir, self.of_video_dir):
            if base is None:
                continue
            name = self.video_pattern.format(vid=tile_id)
            candidate = base / name
            if candidate.exists():
                return candidate
            matches = list(base.glob(_video_glob(self.video_pattern, tile_id)))
            if matches:
                return matches[0]
        return None


    def _get_of_video_path_for(self, tile_id: int) -> Optional[Path]:
        """Resolve the video source used by fresh optical-flow/PIV compute.

        This is intentionally separate from `_get_video_path_for`: the
        display overlay can use lightweight realtime TIFF stacks while OF
        recomputation reads the raw frame folders.
        """
        if self.of_video_dir is None or self.of_video_pattern is None:
            return None
        name = self.of_video_pattern.format(vid=tile_id)
        candidate = self.of_video_dir / name
        if candidate.exists():
            return candidate
        matches = list(self.of_video_dir.glob(_video_glob(self.of_video_pattern, tile_id)))
        if matches:
            return matches[0]
        return None


    def _remove_video_overlay(self):
        # Stop autoplay before destroying anything that the
        # AnimationThread is connected to.
        try:
            self.viewer.window._qt_viewer.dims.stop()
        except Exception:
            pass
        if self._video_layer is not None:
            try:
                self.viewer.layers.remove(self._video_layer)
            except Exception:
                pass
            self._video_layer = None
            self._video_tile_id = None
        # Per-tile harmonic cache is tile-local; without a tile, drop it.
        self._tile_harmonic_cache.clear()


    def _populate_tile_harmonic_cache(self, tile_id: int):
        """Compute per-tile harmonic fits for every edge that has a
        measurement on `tile_id`.  Uses THAT tile's measurement (not
        the best-quality one) so the f0 is consistent everywhere in
        the view.

        Reads raw `measurements_piv` at each tile's native f0.
        """
        self._tile_harmonic_cache.clear()
        meas_key = 'measurements_piv'
        n = 0
        for u, v, d in self.G.edges(data=True):
            meas = d.get(meas_key) or []
            m_tile = next((mm for mm in meas
                           if MeasurementView(mm).tile_id == tile_id), None)
            if m_tile is None:
                continue
            mv_tile = MeasurementView(m_tile)
            if mv_tile.Q_t is None:
                continue
            if not _measurement_usable(m_tile):
                continue
            Q_t = np.asarray(mv_tile.Q_t, dtype=float)
            if Q_t.size < 30:
                continue
            f0_for_fit = float(mv_tile.harm.f0_hz or 2.5)
            full = _harmonic_fit_full(Q_t, f0=f0_for_fit, dt=1.0/250)
            if full is None:
                continue
            self._tile_harmonic_cache[(u, v)] = full
            n += 1
        print(f"  Computed per-tile harmonics for {n} edges in tile "
              f"{tile_id} (f0 anchored at tile-{tile_id}'s value).")


    def _load_video_overlay(self, tile_id: int):
        """Load the per-tile video and add it as a napari 3D Image layer
        translated/scaled onto the mosaic so the embryo lines up with
        the graph.  Replaces any prior overlay (only one at a time).

        Tile switches reuse the same Image layer (in-place data update).
        Destroying and re-adding the layer leaves napari's
        AnimationThread holding a dangling reference to the destroyed
        previous QtDimSliderWidget, which raises TypeError on the next
        .play() (`prev_slider.fps_changed.disconnect(...)` fails).
        Keeping the layer alive avoids that path entirely.
        """
        if tile_id == self._video_tile_id:
            return  # already showing
        # Stop autoplay before mutating layer data — keeps the
        # AnimationThread from sampling .data during the swap.
        try:
            self.viewer.window._qt_viewer.dims.stop()
        except Exception:
            pass
        if tile_id not in self.tiles:
            print(f"  No tile_positions entry for tile {tile_id}; "
                  f"can't place the video on the mosaic.")
            return
        video_path = self._get_video_path_for(tile_id)
        if video_path is None:
            print(f"  No video found for tile {tile_id} "
                  f"(looked under {self.video_dir} with pattern "
                  f"{self.video_pattern!r}).")
            return
        from ..io.tiff import load_tiff_stack, cut_before_fade
        print(f"  Loading video for tile {tile_id}: {video_path}")
        stack = load_tiff_stack(video_path, max_frames=600)
        stack, cut_frame = cut_before_fade(stack, verbose=False)
        # Temporally decimate for ~real-time playback.  The 30-fps play rate
        # (below) assumes the realtime preview is ~8× block-averaged (~75-90
        # frames ≈ real recording duration).  This stage's realtime videos are
        # only ~4× downsampled (~200 frames), so 30 fps ran ~2× too slow.
        # Decimate to the ~90-frame target so playback ≈ real time regardless
        # of how much the preview file was pre-downsampled.
        _pt = 90
        if stack.shape[0] > int(_pt * 1.4):
            _stride = max(1, int(round(stack.shape[0] / _pt)))
            stack = stack[::_stride]
            print(f"  temporally decimated video ×{_stride} → "
                  f"{stack.shape[0]} frames (≈ real-time at 30 fps)")
        # Normalise to uint8 with 1%/99% contrast — keeps GPU upload cheap
        # and gives a sensible default contrast.
        if stack.dtype != np.uint8:
            sample = stack[::max(1, stack.shape[0] // 20)]
            lo, hi = np.percentile(sample, (1.0, 99.0))
            if hi <= lo:
                hi = lo + 1.0
            stack = np.clip(
                (stack.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                0, 255).astype(np.uint8)
        # Compute the effective scale factor between the loaded video's
        # frame size and the canonical raw frame size.  Full-res
        # acquisitions are TILE_RAW_HEIGHT × TILE_RAW_WIDTH; bundles
        # that ship pre-downsampled videos will arrive smaller.  This
        # lets a single overlay path handle both the full-res and the
        # 2×-downsampled bundles correctly — without this detection
        # the bundle videos sit in the top-left quarter of their
        # intended footprint because the viewer's internal `[::2,::2]`
        # below would compound the bundle's downsample.
        H_in, W_in = stack.shape[1], stack.shape[2]
        ds_factor_y = TILE_RAW_HEIGHT / float(H_in) if H_in else 1.0
        ds_factor_x = TILE_RAW_WIDTH / float(W_in) if W_in else 1.0
        # Only run the internal GPU-memory downsample if the input is
        # close to full resolution (within ~30%).  Below that we trust
        # the bundle's downsampling and skip the slicing — the loaded
        # video is already small enough.
        display_downsampled = False
        if ds_factor_x < 1.5 and ds_factor_y < 1.5:
            stack = np.ascontiguousarray(stack[:, ::2, ::2])
            ds_factor_y *= 2.0
            ds_factor_x *= 2.0
            display_downsampled = True
        T, H, W = stack.shape
        # tile_width/height come from the mosaic_graph (we don't have a
        # mosaic-image reference if the user didn't pass --tiff).  Use
        # scale_x/scale_y as a multiplicative correction relative to the
        # raw video pixels.  The placement convention matches
        # `_tile_mgmt._load_tile_video`.
        tile_entry = self.tiles[tile_id]
        A_tile = tile_entry.get('affine')
        # Build a 4×4 spatial affine for the (T, Y, X) image.  Time axis
        # is identity; spatial axes carry the tile's full 2×3 affine
        # (rotation + scale + translate), pre-multiplied by a diagonal
        # scale that maps the *loaded* video's pixel size back to the
        # canonical raw-frame pixel size.  Global tile offset shifts
        # negative translates into image-row coords.
        S_pixel = np.array([[ds_factor_y, 0.0], [0.0, ds_factor_x]])
        if A_tile is not None:
            M = A_tile[:2, :2] @ S_pixel
            t_yx = A_tile[:2, 2] - np.array([self._tile_offset_y,
                                             self._tile_offset_x])
        else:
            M = np.diag([tile_entry['scale_y'] * ds_factor_y,
                         tile_entry['scale_x'] * ds_factor_x])
            t_yx = np.array([tile_entry['translate_y'] - self._tile_offset_y,
                             tile_entry['translate_x'] - self._tile_offset_x])
        affine44 = np.eye(4)
        affine44[1:3, 1:3] = M
        affine44[1:3, 3] = t_yx
        print(f"  video source {H_in}×{W_in}"
              f"{' (display downsampled 2x)' if display_downsampled else ''}"
              f" → pixel-scale ({ds_factor_y:.2f}, {ds_factor_x:.2f})"
              f"  tile_translate=({t_yx[0]:.0f}, {t_yx[1]:.0f})")
        new_name = (f"Video tile {tile_id} "
                    f"({H_in}x{W_in}"
                    f"{' -> display 2x' if display_downsampled else ''})")
        # Try in-place layer reuse first.  If the existing layer is
        # still attached to the viewer we mutate its data + transform
        # without destroying the QtDimSliderWidget that the
        # AnimationThread is connected to.
        replaced_in_place = False
        if (self._video_layer is not None
                and self._video_layer in self.viewer.layers):
            try:
                self._video_layer.data = stack
                self._video_layer.scale = (1.0, 1.0, 1.0)
                self._video_layer.translate = (0.0, 0.0, 0.0)
                self._video_layer.affine = affine44
                self._video_layer.name = new_name
                self._video_layer.contrast_limits = (0, 255)
                replaced_in_place = True
            except Exception as exc:
                print(f"  in-place video swap failed ({exc!r}); "
                      f"falling back to remove+add.")
                self._remove_video_overlay()
        if not replaced_in_place:
            self._video_layer = self.viewer.add_image(
                stack,
                name=new_name,
                colormap='gray',
                contrast_limits=(0, 255),
                interpolation2d='nearest',
                affine=affine44,
                opacity=0.85,
                blending='translucent',
            )
        self._video_tile_id = tile_id
        # Populate per-tile harmonic cache (used by the time-series
        # inspect panel for cross-edge-comparable phase at the loaded
        # tile's f0).  Fast: typically 100–500 edges × ~5 ms each.
        self._populate_tile_harmonic_cache(tile_id)
        # Reset the time slider to frame 0.  Use the modern `current_step`
        # API — `set_point(0, 0)` is deprecated in napari ≥ 0.5 and on a
        # second layer swap can leave the slider at a stale index that
        # points past the end of a shorter new stack → blank display.
        try:
            cs = list(self.viewer.dims.current_step)
            if len(cs) >= 1:
                cs[0] = 0
                self.viewer.dims.current_step = tuple(cs)
        except Exception:
            try:
                self.viewer.dims.set_point(0, 0)  # legacy fallback
            except Exception:
                pass
        # Defer play() to the next event-loop tick so the new layer's
        # data/transform settles before the AnimationThread attaches.
        # Wrap in a defensive scrub: if napari's AnimationThread is
        # still holding a stale prev_slider reference from a prior
        # session (would raise TypeError on fps_changed.disconnect),
        # clear it so play() starts cleanly.
        def _start_play():
            qt_dims = None
            try:
                qt_dims = self.viewer.window._qt_viewer.dims
            except Exception:
                return
            try:
                at = getattr(qt_dims, '_animation_thread', None)
                # napari's set_slider reads prev_slider = self.slider on
                # the AnimationThread and calls
                # prev_slider.fps_changed.disconnect(...) — that raises
                # TypeError if prev_slider was destroyed by an earlier
                # layer removal.  Clearing at.slider to None skips the
                # disconnect path on the first attach to the new widget.
                if at is not None and hasattr(at, 'slider'):
                    at.slider = None
            except Exception:
                pass
            try:
                # 30 fps display rate.  Pair with the realtime preview
                # videos (8× temporally block-averaged) for ~real-time
                # playback: 75 frames / 30 fps ≈ 2.5 s, matching the
                # 2.4 s real recording duration.
                qt_dims.play(axis=0, fps=30)
            except TypeError:
                # Last-resort scrub: napari's internal disconnect chose
                # to raise — swallow it; the new slider will still play
                # on the user's next manual interaction.
                pass
            except Exception:
                pass
        try:
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, _start_play)
        except Exception:
            _start_play()
        cut_note = f" (truncated at frame {cut_frame})" if cut_frame else ""
        print(f"  → {T} frames placed via affine "
              f"translate=({t_yx[0]:.0f}, {t_yx[1]:.0f}){cut_note}")


    def _precompute_measured_dissipation(self):
        """Compute viscous dissipation per edge from PIV Q_t + geometry.

        Φ = r · L · ⟨Q²⟩  (Watts)
        with r = 8μ / (πR⁴), L from edge attr `length`, R from `radius`,
        and ⟨Q²⟩ = mean(Q_t²) using the best PIV measurement.  Result
        stored as `_meas_dissipation` on each edge (in-memory only).
        Edges without a usable measurement / geometry get nan.
        """
        from ..analyze.constants import PX_SIZE_UM
        from ..analyze.constants import MU_DEFAULT
        px_to_m = PX_SIZE_UM * 1e-6
        nl_to_m3 = 1e-12  # 1 nL = 1e-12 m³  → Q [nL/s] × 1e-12 = Q [m³/s]
        n = 0
        for u, v, d in self.G.edges(data=True):
            if '_meas_dissipation' in d:
                continue
            m = _best_measurement(d.get('measurements_piv'))
            mv = MeasurementView(m) if m is not None else None
            R_px = d.get('radius')
            L_px = d.get('length')
            # v4 migration: was m.get('Q_t') is None
            if (mv is None or mv.Q_t is None
                    or R_px is None or L_px is None):
                d['_meas_dissipation'] = float('nan')
                continue
            try:
                R_px_f = float(R_px); L_px_f = float(L_px)
            except (TypeError, ValueError):
                d['_meas_dissipation'] = float('nan')
                continue
            if not (np.isfinite(R_px_f) and R_px_f > 0
                    and np.isfinite(L_px_f) and L_px_f > 0):
                d['_meas_dissipation'] = float('nan')
                continue
            R_m = R_px_f * px_to_m
            L_m = L_px_f * px_to_m
            r = 8.0 * MU_DEFAULT / (np.pi * R_m ** 4)
            Q_t = np.asarray(mv.Q_t, dtype=float) * nl_to_m3  # v4: was m['Q_t']
            Q2 = float(np.nanmean(Q_t ** 2))
            d['_meas_dissipation'] = r * L_m * Q2
            n += 1
        print(f"  Precomputed dissipation for {n} edges "
              f"(Φ in W; range typical 1e-15 to 1e-12)")


    def _precompute_harmonic_classes(self):
        """Run one harmonic fit per edge (DC + H1 + H2 + H3) and cache
        everything needed for the per-harmonic field views:

          • `harmonic_class`   — int 0/1/2/3 or None
          • `_h_amp_DC`, `_h_Z_DC`
          • `_h_amp_H1`, `_h_phase_H1`, `_h_Z_H1`
          • `_h_amp_H2`, `_h_phase_H2`, `_h_Z_H2`
          • `_h_amp_H3`, `_h_phase_H3`, `_h_Z_H3`
          • `_h_r2`, `_h_sigma`

        The `_h_` prefix marks them as viewer-cached derived attributes.
        Cached back to the gpickle if `cache_harmonic_class` is True so
        relaunches are instant.

        Skip rules:
          - If `harmonic_class` is already present on every edge and
            the user didn't pass `--force-recompute-harmonic-class`,
            do nothing.
          - If `--force-recompute-harmonic-class`, wipe existing cache
            first and recompute from scratch.
        """
        import time
        CACHED_KEYS = (
            'harmonic_class',
            '_h_amp_DC', '_h_Z_DC',
            '_h_amp_H1', '_h_phase_H1', '_h_Z_H1',
            '_h_amp_H2', '_h_phase_H2', '_h_Z_H2',
            '_h_amp_H3', '_h_phase_H3', '_h_Z_H3',
            '_h_r2', '_h_sigma',
            '_h_total_snr', '_h_cache_ver',
        )
        if self.force_recompute_harmonic_class:
            for _, _, d in self.G.edges(data=True):
                for k in CACHED_KEYS:
                    d.pop(k, None)
        # Skip the recompute only if EVERY edge has the full `_h_*` cache
        # AT THE CURRENT SCHEMA VERSION.  Bumping HARMONIC_CACHE_VERSION
        # invalidates everything stored under the old convention (e.g.
        # σ definition changed std → MAD) without needing the user to
        # pass --force-recompute-harmonic-class.
        n_with_class = 0
        n_with_h1_amp = 0
        n_with_cur_ver = 0
        for _, _, d in self.G.edges(data=True):
            if 'harmonic_class' in d:
                n_with_class += 1
            if '_h_amp_H1' in d:
                n_with_h1_amp += 1
            if d.get('_h_cache_ver', 0) >= HARMONIC_CACHE_VERSION:
                n_with_cur_ver += 1
        n_edges = self.G.number_of_edges()
        if (n_with_class == n_edges
                and n_with_h1_amp == n_edges
                and n_with_cur_ver == n_edges):
            print(f"  harmonic_class + `_h_*` cache present on all "
                  f"{n_with_class} edges at version "
                  f"{HARMONIC_CACHE_VERSION}; skipping precompute.")
            return
        if n_with_class == n_edges and n_with_cur_ver < n_edges:
            print(f"  harmonic cache present but at older schema "
                  f"version (have {n_with_cur_ver}/{n_edges} at v"
                  f"{HARMONIC_CACHE_VERSION}); recomputing with new "
                  f"MAD-σ + total_snr convention.")
            # Wipe stale entries so the loop below actually re-fits
            # them (otherwise the `_h_amp_H1 in d` skip would fire).
            for _, _, d in self.G.edges(data=True):
                if d.get('_h_cache_ver', 0) < HARMONIC_CACHE_VERSION:
                    for k in CACHED_KEYS:
                        d.pop(k, None)
        elif n_with_class == n_edges:
            print(f"  harmonic_class present on all {n_with_class} edges "
                  f"but `_h_*` cache missing on {n_edges - n_with_h1_amp}; "
                  f"running full precompute …")
        n_total = self.G.number_of_edges()
        n_resolved = 0
        n_failed = 0
        t0 = time.time()
        print(f"  Precomputing harmonic_class for {n_total} edges ...")
        for i, (u, v, d) in enumerate(self.G.edges(data=True)):
            # Skip only if BOTH the class label AND the extended
            # `_h_*` cache are present — otherwise we re-fit so the
            # 4-selector resolver can read amp/phase/Z at H₁/H₂/H₃.
            if 'harmonic_class' in d and '_h_amp_H1' in d:
                continue
            m = _best_measurement(d.get('measurements_piv'))
            mv = MeasurementView(m) if m is not None else None
            # v4 migration: was m.get('Q_t') / m.get('f0_hz')
            if m is None or not _measurement_usable(m) \
                    or mv.Q_t is None:
                d['harmonic_class'] = None
                continue
            Q_t = np.asarray(mv.Q_t, dtype=float)
            if Q_t.size < 30:
                d['harmonic_class'] = None
                continue
            full = _harmonic_fit_full(
                Q_t, f0=float(mv.harm.f0_hz or 2.5), dt=1.0/250)
            if full is None:
                d['harmonic_class'] = None
                n_failed += 1
                continue
            # Pack Z dict for the classification helper.
            snrs = {'DC': full['Z_DC'], 'H1': full['Z_H1'],
                    'H2': full['Z_H2'], 'H3': full['Z_H3']}
            d['harmonic_class'] = int(_harmonic_class(snrs))
            d['_h_amp_DC'] = full['amp_DC']
            d['_h_Z_DC'] = full['Z_DC']
            for k in (1, 2, 3):
                d[f'_h_amp_H{k}'] = full[f'amp_H{k}']
                d[f'_h_phase_H{k}'] = full[f'phase_H{k}']
                d[f'_h_Z_H{k}'] = full[f'Z_H{k}']
            d['_h_r2'] = full['r2']
            d['_h_sigma'] = full['sigma']
            d['_h_total_snr'] = full.get('total_snr', float('nan'))
            d['_h_cache_ver'] = HARMONIC_CACHE_VERSION
            n_resolved += 1
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-3)
                eta = (n_total - i - 1) / max(rate, 1e-3)
                print(f"    {i + 1}/{n_total} edges  "
                      f"({rate:.0f}/s, ~{eta:.0f}s remaining)")
        elapsed = time.time() - t0
        # Tally distribution
        counts = {0: 0, 1: 0, 2: 0, 3: 0, None: 0}
        for _, _, d in self.G.edges(data=True):
            counts[d.get('harmonic_class')] = (
                counts.get(d.get('harmonic_class'), 0) + 1)
        print(f"  Done in {elapsed:.1f} s.  "
              f"Class distribution: "
              f"DC-only={counts[0]}, +H1={counts[1]}, "
              f"+H1+H2={counts[2]}, +H1+H2+H3={counts[3]}, "
              f"no-data={counts[None]}")
        # Persist back to the gpickle so the next launch skips compute.
        if self.cache_harmonic_class:
            self._save_graph_with_harmonic_class()


    def _save_graph_with_harmonic_class(self):
        """Atomically write self.G back to self.graph_path so the
        harmonic_class attributes survive across viewer launches.

        Atomic = pickle to a temp file in the same directory, then
        os.replace() onto the target.  This guarantees a partial write
        can never leave the gpickle in a corrupt state.

        Only `harmonic_class` is ever added by this viewer; every other
        attribute on G is left as it was loaded.  No edges are added or
        removed.
        """
        if not self.workbench:
            return
        import os, pickle, tempfile
        target = Path(self.graph_path).resolve()
        if not target.parent.is_dir():
            print(f"  Save skipped: parent dir {target.parent} not found.")
            return
        try:
            fd, tmp = tempfile.mkstemp(
                suffix='.harm_save.tmp', dir=str(target.parent),
                prefix='.' + target.stem + '.')
            try:
                with os.fdopen(fd, 'wb') as f:
                    self._dump_graph(f,
                                 protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, target)
                size_mb = target.stat().st_size / (1024 ** 2)
                print(f"  Cached harmonic_class to {target.name} "
                      f"({size_mb:.0f} MB).  "
                      f"Disable with --no-cache-harmonic-class.")
            except Exception:
                try: os.remove(tmp)
                except Exception: pass
                raise
        except Exception as e:
            print(f"  Warning: could not write cache: {e}  "
                  f"(compute will repeat next launch)")


    def _discover_fields(self) -> List[str]:
        """Keep only the fields actually present in this graph's data."""
        found: List[str] = []
        # Top-level edge attributes (geometry + precomputed harmonic_class).
        for f in ('radius', 'length', 'harmonic_class'):
            for _, _, d in self.G.edges(data=True):
                if f in d:
                    found.append(f); break
        # Dissipation lives under `_meas_dissipation` (precomputed at
        # load time); surface it as the user-facing `dissipation` field.
        for _, _, d in self.G.edges(data=True):
            if '_meas_dissipation' in d:
                found.append('dissipation'); break
        # PIV-side fields
        for f in DEFAULT_FIELDS:
            if f in found:
                continue
            for _, _, d in self.G.edges(data=True):
                piv = d.get('measurements_piv') or []
                if any(f in m for m in piv):
                    found.append(f); break
        # Categorical PIV fields (quality_tier).  harmonic_class is also
        # categorical but lives top-level, so it's already in `found`.
        for f in CATEGORICAL_FIELDS:
            if f in found:
                continue
            for _, _, d in self.G.edges(data=True):
                piv = d.get('measurements_piv') or []
                if any(f in m for m in piv):
                    found.append(f); break
        # When a sim has been run, expose the sim-only fields — these
        # have no measured equivalent.  Other sim outputs (mean_Q,
        # amp_Q, PI, phase) share names with measured and route via
        # the Source toggle, not the dropdown.
        if getattr(self, '_sim_active', False):
            sim_only_checks = [
                ('pressure_mean',   '_sim_tmp_pressure_mean'),
                ('pressure_amp',    '_sim_tmp_pressure_amp'),
                ('pressure_amp_H2', '_sim_tmp_pressure_amp_H2'),
                ('pressure_phase',  '_sim_tmp_phase_P'),
                ('pressure_drop',   '_sim_tmp_pressure_drop'),
            ]
            for display_name, edge_key in sim_only_checks:
                for _, _, d in self.G.edges(data=True):
                    if edge_key in d:
                        found.append(display_name); break
        return found


    # ── napari + Qt UI ─────────────────────────────────────────────────
    def _setup_viewer(self):
        import napari
        title = f"Mosaic (read-only) — {self.graph_path.name}"
        self.viewer = napari.Viewer(title=title)

        # Background image (if available)
        if self.mosaic is not None:
            self.viewer.add_image(self.mosaic, name='Mosaic', opacity=0.75)
            # Optional vessel-mask overlay from the stage's mask/mask.tif
            # (same fused-pixel grid as the mosaic).  Hidden by default —
            # toggle its visibility in napari's layer list.
            try:
                if self.tiff_path is not None:
                    _mask_p = self.tiff_path.parent.parent / 'mask' / 'mask.tif'
                    if _mask_p.exists():
                        import tifffile as _tf
                        _m = _tf.imread(str(_mask_p))
                        if _m.ndim == 2 and _m.shape == self.mosaic.shape[:2]:
                            self.viewer.add_labels(
                                (_m > 0).astype('uint8'),
                                name='Vessel mask', opacity=0.4, visible=False)
                            print(f"  mask overlay: {_mask_p.name} "
                                  f"({int((_m > 0).sum())} px, toggle in layer list)")
            except Exception as _e:
                print(f"  (mask overlay skipped: {_e})")

        # Edges layer is built lazily by _refresh_edges (which removes the
        # previous one and re-adds a fresh one).  This sidesteps a napari
        # quirk where Shapes layers don't re-render colour after a bulk
        # .data + .edge_color update on an initially-empty layer.
        self._edges_layer = None
        # Nodes layer (hidden by default). napari ≥ 0.5 renamed edge_* →
        # border_*; keep an older-napari fallback for old envs.
        try:
            self._nodes_layer = self.viewer.add_points(
                np.empty((0, 2)), size=4, face_color='white',
                border_color='black', name='Nodes', visible=False,
            )
        except TypeError:
            self._nodes_layer = self.viewer.add_points(
                np.empty((0, 2)), size=4, face_color='white',
                edge_color='black', name='Nodes', visible=False,
            )
        # Comparison-set marker layer — large halo rings, per-point
        # coloured to match the comparison plot.  Sized larger than the
        # cyan click marker so both stay visible when the user clicks an
        # edge that's already in the comparison set.
        try:
            self._comparison_marker_layer = self.viewer.add_points(
                np.empty((0, 2)), size=36, face_color='transparent',
                border_color='#1f77b4', border_width=0.35,
                name='Comparison set', symbol='ring', opacity=0.95,
            )
        except TypeError:
            self._comparison_marker_layer = self.viewer.add_points(
                np.empty((0, 2)), size=36, face_color='transparent',
                edge_color='#1f77b4', edge_width=0.35,
                name='Comparison set', symbol='ring', opacity=0.95,
            )

        # Click-highlight marker.  napari ≥ 0.5 renamed edge_* → border_*
        # AND made border_width relative to point size by default (0-1).
        # Pass a relative width (~20% of size) which works on both old
        # and new napari API names; if your napari is too old to accept
        # `border_width`, swap it for `edge_width=0.2`.
        try:
            self._click_marker_layer = self.viewer.add_points(
                np.empty((0, 2)), size=24, face_color='transparent',
                border_color='cyan', border_width=0.2,
                name='Selected', symbol='ring',
            )
        except TypeError:
            self._click_marker_layer = self.viewer.add_points(
                np.empty((0, 2)), size=24, face_color='transparent',
                edge_color='cyan', edge_width=0.2,
                name='Selected', symbol='ring',
            )

        # Boundary-node badges — red dot + "A" at arterial sources, blue
        # dot + "V" at venous sinks.  Always visible above the edges.
        def _to_display(n):
            return [self.mosaic_height - self.G.nodes[n]['y'],
                    self.G.nodes[n]['x']]
        if self._source_nodes:
            src_pts = np.array([_to_display(n) for n in self._source_nodes])
            self._source_layer = self.viewer.add_points(
                src_pts, size=18, face_color='#d62728',
                border_color='white',
                name='Arterial boundary (A)',
                text={'string': ['A'] * len(self._source_nodes),
                      'size': 12, 'color': 'white',
                      'anchor': 'center'},
            )
        if self._sink_nodes:
            snk_pts = np.array([_to_display(n) for n in self._sink_nodes])
            self._sink_layer = self.viewer.add_points(
                snk_pts, size=18, face_color='#1f77b4',
                border_color='white',
                name='Venous boundary (V)',
                text={'string': ['V'] * len(self._sink_nodes),
                      'size': 12, 'color': 'white',
                      'anchor': 'center'},
            )

        # Click handler attached after every layer rebuild via
        # `_attach_edge_click`.
        # Use the viewer-wide mouse-drag callback so it survives layer
        # rebuilds: it looks up the click position and asks
        # _inspect_nearest_edge to do hit testing against our
        # precomputed midpoints.  Boundary-node hits are intercepted
        # first so the user can inspect BC harmonics.
        @self.viewer.mouse_drag_callbacks.append
        def _on_click(viewer, event):
            if event.type != 'mouse_press' or len(event.position) < 2:
                return
            # Region tag/untag armed: clicks belong to the shape being
            # drawn.  Inspecting here would rebuild the highlight layer,
            # which napari makes the active layer — stealing the clicks
            # from the region layer after a couple of presses.
            _rl = getattr(self, '_region_tag_layer', None)
            if (_rl is not None and _rl in self.viewer.layers
                    and str(_rl.mode).startswith('add_')):
                return
            row, col = self._world_to_mosaic_data(event.position)
            # Edit-tab boundary marking takes priority — when a mode is
            # active, every click is a node-tag action.
            if getattr(self, '_edit_mode', None):
                self._on_edit_click(col, self.mosaic_height - row)
                return
            if self._inspect_nearest_boundary_node(col, row):
                return
            # Click-to-tag: bypass the info-inspect flow and toggle the
            # nearest edge's radial_uncertain flag directly.  Same
            # hit-testing radius as the normal inspect click.
            if getattr(self, '_click_to_tag_mode', False):
                self._tag_nearest_edge(col, row)
                return
            self._inspect_nearest_edge(col, row)


    def _world_to_mosaic_data(self, world_pos):
        """Convert a napari world position (event.position) to mosaic
        data coordinates (row, col).

        With no rotation active, world == data for the Mosaic layer
        (identity affine), so this just returns the last two elements
        of world_pos.  With --rotate-cw/--rotate-ccw, the Mosaic layer
        carries an affine whose inverse maps the clicked world position
        back to the unrotated mosaic row/col — that's where the graph
        node coords live, so hit-testing against `_edge_midpoints`
        works correctly.
        """
        ref_layer = None
        for name in ('Mosaic', 'Nodes', 'Edges'):
            if name in self.viewer.layers:
                ref_layer = self.viewer.layers[name]
                break
        if ref_layer is None:
            return float(world_pos[-2]), float(world_pos[-1])
        try:
            data_pos = ref_layer.world_to_data(world_pos)
        except Exception:
            return float(world_pos[-2]), float(world_pos[-1])
        return float(data_pos[-2]), float(data_pos[-1])


    def _setup_panel(self):
        from qtpy.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
            QCheckBox, QPushButton, QGroupBox, QTextEdit, QTabWidget,
            QDoubleSpinBox, QScrollArea, QLineEdit,
        )
        # Radial-uncertainty state defaults — must exist before the View
        # tab's uncertainty checkbox reads them.
        if not hasattr(self, '_show_radial_uncertain'):
            self._show_radial_uncertain = False   # off by default
        if not hasattr(self, '_radial_tags_dirty'):
            self._radial_tags_dirty = False
        if not hasattr(self, '_click_to_tag_mode'):
            self._click_to_tag_mode = False
        if not hasattr(self, '_region_tag_layer'):
            self._region_tag_layer = None
        if not hasattr(self, '_region_action_pending'):
            self._region_action_pending = 'tag'
        if not hasattr(self, '_region_shape_pending'):
            self._region_shape_pending = 'add_rectangle'
        tabs = QTabWidget()
        view_panel = QWidget(); view_layout = QVBoxLayout(view_panel)
        ts_panel = QWidget();   ts_layout = QVBoxLayout(ts_panel)
        sim_panel = QWidget();  sim_layout = QVBoxLayout(sim_panel)

        # ══════ Tab 1: VIEW ═══════════════════════════════════════════
        layout = view_layout    # alias for the existing widget code

        # ── Data source: Measured (PIV) / Reconciled (conservative DC + aligned AC) / Simulated ──
        from qtpy.QtWidgets import QRadioButton, QButtonGroup
        src_group = QGroupBox("Data source")
        src_lay = QHBoxLayout(src_group)
        self._src_radio_measured = QRadioButton("Measured")
        self._src_radio_measured.setChecked(self.current_source == 'measured')
        self._src_radio_rec = QRadioButton("Reconciled")
        self._src_radio_rec.setChecked(self.current_source == 'reconciled')
        self._src_radio_rec.setEnabled(self._reconciliation_available())
        self._src_radio_rec.setToolTip(
            "Conservative DC flow Q\u2032 and tile-aligned harmonics (results/dc_reconciled_v2.pkl, ac_field_v2.pkl)."
            if self._reconciliation_available() else
            "No reconciliation found: run scripts/dc_reconcile.py and scripts/ac_field.py for this stage.")
        self._src_radio_sim = QRadioButton("Sim (intact)")
        self._src_radio_sim.setChecked(self.current_source == 'sim')
        self._src_radio_sim.setEnabled(self._sim_active)
        if not self._sim_active:
            self._src_radio_sim.setToolTip(
                "Run a simulation in the Simulate tab to enable.")
        # Carved sim option — enabled once any `_h_*_sim_carved` fields exist
        # on the graph (populated by the Carved Sim tab's Run button).
        self._src_radio_sim_carved = QRadioButton("Sim (carved)")
        self._src_radio_sim_carved.setChecked(
            self.current_source == 'sim_carved')
        self._src_radio_sim_carved.setEnabled(
            self._carved_sim_data_available())
        if not self._src_radio_sim_carved.isEnabled():
            self._src_radio_sim_carved.setToolTip(
                "Run the porous-carved sim in the Carved Sim tab to enable.")
        self._src_button_group = QButtonGroup(src_group)
        self._src_button_group.addButton(self._src_radio_measured)
        self._src_button_group.addButton(self._src_radio_rec)
        self._src_button_group.addButton(self._src_radio_sim)
        self._src_button_group.addButton(self._src_radio_sim_carved)
        self._src_radio_measured.toggled.connect(self._on_source_change)
        self._src_radio_rec.toggled.connect(self._on_source_change)
        self._src_radio_sim.toggled.connect(self._on_source_change)
        self._src_radio_sim_carved.toggled.connect(self._on_source_change)
        src_lay.addWidget(self._src_radio_measured)
        src_lay.addWidget(self._src_radio_rec)
        src_lay.addWidget(self._src_radio_sim)
        src_lay.addWidget(self._src_radio_sim_carved)
        src_lay.addStretch(1)
        layout.addWidget(src_group)

        # Field selector — four small combos compose the colormap.
        fld_group = QGroupBox("Color edges by")
        fld_lay = QVBoxLayout(fld_group)

        # Quantity row (Q / P; P disabled in Measured source)
        q_row = QHBoxLayout()
        q_row.addWidget(QLabel('Quantity:'))
        self._quantity_combo = QComboBox()
        self._quantity_combo.addItem('Q (flow)',     userData='Q')
        self._quantity_combo.addItem('P (pressure)', userData='P')
        self._quantity_combo.setCurrentIndex(
            0 if self.current_quantity == 'Q' else 1)
        self._quantity_combo.currentIndexChanged.connect(
            self._on_quantity_change)
        q_row.addWidget(self._quantity_combo); q_row.addStretch(1)
        fld_lay.addLayout(q_row)

        # Property row
        p_row = QHBoxLayout()
        p_row.addWidget(QLabel('Property:'))
        self._property_combo = QComboBox()
        for key, label, _is_harm, _valid_in in PROPERTY_DEFS:
            self._property_combo.addItem(label, userData=key)
            if key == self.current_property:
                self._property_combo.setCurrentIndex(
                    self._property_combo.count() - 1)
        self._property_combo.currentIndexChanged.connect(
            self._on_property_change)
        p_row.addWidget(self._property_combo); p_row.addStretch(1)
        fld_lay.addLayout(p_row)

        # Harmonic row (only enabled for harmonic-keyed properties)
        h_row = QHBoxLayout()
        h_row.addWidget(QLabel('Harmonic:'))
        self._harmonic_combo = QComboBox()
        for hk in HARMONIC_KEYS:
            self._harmonic_combo.addItem(HARMONIC_LABELS[hk], userData=hk)
            if hk == self.current_harmonic:
                self._harmonic_combo.setCurrentIndex(
                    self._harmonic_combo.count() - 1)
        self._harmonic_combo.currentIndexChanged.connect(
            self._on_harmonic_change)
        h_row.addWidget(self._harmonic_combo); h_row.addStretch(1)
        fld_lay.addLayout(h_row)

        # Sync enable-state of the combos (Quantity, Harmonic) and the
        # item flags inside Property to the current Source + selection.
        self._sync_field_combos()

        self._log_chk = QCheckBox("Log scale")
        self._log_chk.setToolTip("Useful for mean_Q, amp_Q (orders of magnitude).")
        self._log_chk.toggled.connect(self._on_log_toggle)
        fld_lay.addWidget(self._log_chk)

        # ── Radial-uncertainty tagging ──────────────────────────────
        # Small panel in the View tab so the user can visually mark
        # vessels whose radius they don't trust, without touching the
        # solver.  The flag persists as ``edge['radial_uncertain']=True``
        # on the graph — downstream analysis can filter on it.
        _ru_row = QWidget()
        _ru_lay = QVBoxLayout(_ru_row)
        _ru_lay.setContentsMargins(0, 4, 0, 0)
        self._radial_uncertain_chk = QCheckBox(
            "Highlight uncertain vessels (orange)")
        self._radial_uncertain_chk.setChecked(self._show_radial_uncertain)
        self._radial_uncertain_chk.setToolTip(
            "When on, edges flagged ``radial_uncertain=True`` render in "
            "orange regardless of the current colormap.  Toggle 'u' on a "
            "selected edge to add/remove the flag; press Save to persist.")
        self._radial_uncertain_chk.toggled.connect(
            self._on_radial_uncertain_toggle)
        _ru_lay.addWidget(self._radial_uncertain_chk)

        # Click-to-tag mode: while ON, every click on a vessel toggles
        # its tag instead of showing info.  Much faster for bulk tagging.
        self._click_to_tag_chk = QCheckBox("Click-to-tag mode")
        self._click_to_tag_chk.setChecked(False)
        self._click_to_tag_chk.setToolTip(
            "When on, clicking a vessel directly toggles its "
            "``radial_uncertain`` flag instead of opening the info box.  "
            "Turn off to return to normal inspect-on-click behaviour.")
        self._click_to_tag_chk.toggled.connect(
            self._on_click_to_tag_toggle)
        _ru_lay.addWidget(self._click_to_tag_chk)

        _btn_row = QWidget()
        self._radial_btn_row = _btn_row
        _btn_lay = QHBoxLayout(_btn_row)
        _btn_lay.setContentsMargins(0, 0, 0, 0)
        self._radial_toggle_btn = QPushButton("Toggle on selected  (u)")
        self._radial_toggle_btn.setToolTip(
            "Flip the ``radial_uncertain`` flag on the currently selected "
            "edge.  Same as pressing 'u' in the napari canvas.")
        self._radial_toggle_btn.clicked.connect(
            self._toggle_radial_uncertain_on_selected)
        _btn_lay.addWidget(self._radial_toggle_btn)
        self._radial_save_btn = QPushButton("Save tags")
        self._radial_save_btn.setToolTip(
            "Write the current radial_uncertain flags back to the graph "
            "file on disk.  No solver rerun.")
        self._radial_save_btn.clicked.connect(self._save_radial_uncertain_tags)
        _btn_lay.addWidget(self._radial_save_btn)
        _ru_lay.addWidget(_btn_row)

        # Region-based tag/untag: pick a shape (Rectangle or Lasso), then
        # press Tag or Untag — every edge midpoint inside the drawn region
        # gets ``radial_uncertain`` flipped in that direction.  ONE scratch
        # Shapes layer is reused across sessions so we never hit the
        # napari layer-lifecycle race that made the old "Rectangle tag…"
        # button stop working after ~two uses.
        _rg_row = QWidget()
        self._region_row = _rg_row
        _rg_lay = QHBoxLayout(_rg_row)
        _rg_lay.setContentsMargins(0, 0, 0, 0)
        _rg_lay.addWidget(QLabel("Region:"))
        self._region_shape_combo = QComboBox()
        self._region_shape_combo.addItem("Rectangle", userData="add_rectangle")
        self._region_shape_combo.addItem("Ellipse", userData="add_ellipse")
        self._region_shape_combo.addItem("Lasso", userData="add_polygon_lasso")
        self._region_shape_combo.addItem("Polygon", userData="add_polygon")
        self._region_shape_combo.setToolTip(
            "Rectangle / Ellipse: drag.  Lasso: click, trace, click to "
            "close.  Polygon: click corners, double-click to close.")
        _rg_lay.addWidget(self._region_shape_combo)
        self._region_tag_btn = QPushButton("Tag")
        self._region_tag_btn.setToolTip(
            "Draw the selected region on the canvas — every edge whose "
            "midpoint falls inside gets ``radial_uncertain=True`` on release.")
        self._region_tag_btn.clicked.connect(
            lambda: self._start_region_action("tag"))
        _rg_lay.addWidget(self._region_tag_btn)
        self._region_untag_btn = QPushButton("Untag")
        self._region_untag_btn.setToolTip(
            "Draw a region — every edge midpoint inside gets its "
            "``radial_uncertain`` flag CLEARED on release.")
        self._region_untag_btn.clicked.connect(
            lambda: self._start_region_action("untag"))
        _rg_lay.addWidget(self._region_untag_btn)
        self._region_stop_btn = QPushButton("Stop")
        self._region_stop_btn.setToolTip(
            "Leave region drawing and return to normal click behaviour.")
        self._region_stop_btn.clicked.connect(self._reset_region_layer)
        _rg_lay.addWidget(self._region_stop_btn)
        _rg_lay.addStretch(1)
        _ru_lay.addWidget(_rg_row)

        self._radial_status_lbl = QLabel("")
        self._radial_status_lbl.setStyleSheet(
            "color: #aaaaaa; font-size: 11px;")
        _ru_lay.addWidget(self._radial_status_lbl)

        fld_lay.addWidget(_ru_row)
        # Initial status update.
        self._refresh_radial_status()

        # ── Flow-direction mismatch highlight ───────────────────────
        # Edges measured in ≥2 tiles whose flow DIRECTION (flow_from→
        # flow_to) disagrees between tiles, counting only measurements
        # above the SNR threshold so noise-floor sign flips (where the
        # direction is unidentifiable) don't register.  Read-only —
        # computed from measurements, nothing written to the graph.
        self._show_flow_mismatch = getattr(self, '_show_flow_mismatch', False)
        _fm_row = QWidget()
        _fm_lay = QVBoxLayout(_fm_row)
        _fm_lay.setContentsMargins(0, 4, 0, 0)
        self._flow_mismatch_chk = QCheckBox(
            "Highlight flow-direction mismatches (magenta)")
        self._flow_mismatch_chk.setChecked(self._show_flow_mismatch)
        self._flow_mismatch_chk.setToolTip(
            "Edges measured in ≥2 tiles whose flow direction "
            "(flow_from→flow_to) disagrees between tiles, counting only "
            "measurements with SNR_H1 ≥ the threshold below.  Renders in "
            "magenta regardless of the current colormap.")
        self._flow_mismatch_chk.toggled.connect(self._on_flow_mismatch_toggle)
        _fm_lay.addWidget(self._flow_mismatch_chk)

        _fm_snr_row = QWidget()
        _fm_snr_lay = QHBoxLayout(_fm_snr_row)
        _fm_snr_lay.setContentsMargins(0, 0, 0, 0)
        _fm_snr_lay.addWidget(QLabel("min SNR_H1 (dB):"))
        self._flow_mismatch_snr_spin = QDoubleSpinBox()
        self._flow_mismatch_snr_spin.setRange(-20.0, 40.0)
        self._flow_mismatch_snr_spin.setDecimals(1)
        self._flow_mismatch_snr_spin.setSingleStep(1.0)
        self._flow_mismatch_snr_spin.setValue(3.0)
        self._flow_mismatch_snr_spin.setToolTip(
            "Only measurements with snr_harm_fit_db ≥ this value count "
            "toward a mismatch.  3 dB ≈ 'reliable'; 6 dB ≈ 'strong'.")
        self._flow_mismatch_snr_spin.valueChanged.connect(
            self._on_flow_mismatch_snr_change)
        _fm_snr_lay.addWidget(self._flow_mismatch_snr_spin)
        _fm_snr_lay.addStretch(1)
        _fm_lay.addWidget(_fm_snr_row)

        self._flow_mismatch_status_lbl = QLabel("")
        self._flow_mismatch_status_lbl.setStyleSheet(
            "color: #aaaaaa; font-size: 11px;")
        _fm_lay.addWidget(self._flow_mismatch_status_lbl)
        fld_lay.addWidget(_fm_row)
        self._compute_flow_mismatch_edges()   # initial status + cache

        # Colour-bound controls — two modes selectable by the combo:
        #   'Percentile' (default): clip to the [lo, hi] percentiles of
        #       the finite per-edge distribution.  Loose clip = 2/98.
        #   'Manual':    use the user-typed [vmin, vmax] literally.
        # Phase fields ignore both and stay fixed at [-pi, pi]; diverging
        # fields under Percentile become symmetric, under Manual respect
        # the user's exact limits.
        bounds_row = QWidget()
        b_lay = QHBoxLayout(bounds_row); b_lay.setContentsMargins(0, 0, 0, 0)
        b_lay.addWidget(QLabel('bounds:'))
        self._bounds_mode_combo = QComboBox()
        self._bounds_mode_combo.addItems(['Percentile', 'Manual'])
        self._bounds_mode_combo.setToolTip(
            "Percentile: colormap clipped to [lo pct, hi pct] of the\n"
            "  finite per-edge distribution (auto-adaptive).\n"
            "Manual: use the exact vmin / vmax typed to the right.\n"
            "Phase fields always stay at [-pi, pi] regardless.")
        self._bounds_mode_combo.currentIndexChanged.connect(
            self._on_bounds_mode_change)
        b_lay.addWidget(self._bounds_mode_combo)
        # Percentile spinboxes (visible when mode == 'Percentile')
        self._pct_lo_lbl = QLabel(' lo pct:')
        b_lay.addWidget(self._pct_lo_lbl)
        self._pct_lo_spin = QDoubleSpinBox()
        self._pct_lo_spin.setRange(0.0, 100.0)
        self._pct_lo_spin.setDecimals(1)
        self._pct_lo_spin.setSingleStep(0.5)
        self._pct_lo_spin.setValue(2.0)
        self._pct_lo_spin.setToolTip(
            "Lower percentile of finite per-edge values, mapped to the\n"
            "low end of the colormap.  Values below get clipped.")
        self._pct_lo_spin.valueChanged.connect(self._on_bounds_change)
        b_lay.addWidget(self._pct_lo_spin)
        self._pct_hi_lbl = QLabel(' hi pct:')
        b_lay.addWidget(self._pct_hi_lbl)
        self._pct_hi_spin = QDoubleSpinBox()
        self._pct_hi_spin.setRange(0.0, 100.0)
        self._pct_hi_spin.setDecimals(1)
        self._pct_hi_spin.setSingleStep(0.5)
        self._pct_hi_spin.setValue(98.0)
        self._pct_hi_spin.setToolTip(
            "Upper percentile of finite per-edge values, mapped to the\n"
            "high end of the colormap.  Values above get clipped.")
        self._pct_hi_spin.valueChanged.connect(self._on_bounds_change)
        b_lay.addWidget(self._pct_hi_spin)
        # Manual vmin / vmax spinboxes (visible when mode == 'Manual')
        self._vmin_lbl = QLabel(' vmin:')
        b_lay.addWidget(self._vmin_lbl)
        self._vmin_spin = QDoubleSpinBox()
        self._vmin_spin.setRange(-1e12, 1e12)
        self._vmin_spin.setDecimals(4)
        self._vmin_spin.setSingleStep(0.1)
        self._vmin_spin.setValue(0.0)
        self._vmin_spin.setToolTip(
            "Manual lower colormap bound (in the field's own units).\n"
            "Under 'Manual' mode this is used exactly, including for\n"
            "diverging fields (limits are NOT auto-symmetrised).")
        self._vmin_spin.valueChanged.connect(self._on_bounds_change)
        b_lay.addWidget(self._vmin_spin)
        self._vmax_lbl = QLabel(' vmax:')
        b_lay.addWidget(self._vmax_lbl)
        self._vmax_spin = QDoubleSpinBox()
        self._vmax_spin.setRange(-1e12, 1e12)
        self._vmax_spin.setDecimals(4)
        self._vmax_spin.setSingleStep(0.1)
        self._vmax_spin.setValue(1.0)
        self._vmax_spin.setToolTip("Manual upper colormap bound.")
        self._vmax_spin.valueChanged.connect(self._on_bounds_change)
        b_lay.addWidget(self._vmax_spin)
        b_lay.addStretch(1)
        fld_lay.addWidget(bounds_row)
        # Start on Percentile mode: hide the manual widgets inline
        # (can't call _on_bounds_mode_change here — the cbar overlay
        # and edge/node layers don't exist yet).
        for _w in (self._vmin_lbl, self._vmin_spin,
                    self._vmax_lbl, self._vmax_spin):
            _w.setVisible(False)

        # Phase-only offset — add a constant to every displayed phase
        # value (wraps into [-180, 180] deg) so you can visually align
        # simulated vs normalized/measured phases.  Ignored for non-
        # phase fields.
        phase_row = QWidget()
        pr_lay = QHBoxLayout(phase_row); pr_lay.setContentsMargins(0, 0, 0, 0)
        pr_lay.addWidget(QLabel('Phase offset (deg):'))
        self._phase_offset_spin = QDoubleSpinBox()
        self._phase_offset_spin.setRange(-360.0, 360.0)
        self._phase_offset_spin.setDecimals(1)
        self._phase_offset_spin.setSingleStep(5.0)
        self._phase_offset_spin.setValue(0.0)
        self._phase_offset_spin.setToolTip(
            "Add a constant offset (in degrees) to every displayed\n"
            "phase value before colour-mapping.  Wraps into [-180, 180].\n"
            "Only active when Property='phase'; ignored for other fields.\n"
            "Use to visually align simulated vs measured/normalized phases.")
        self._phase_offset_spin.valueChanged.connect(self._on_bounds_change)
        pr_lay.addWidget(self._phase_offset_spin)
        _reset_btn = QPushButton('reset')
        _reset_btn.setFixedWidth(60)
        _reset_btn.setToolTip('Reset the phase offset to 0.')
        _reset_btn.clicked.connect(lambda: self._phase_offset_spin.setValue(0.0))
        pr_lay.addWidget(_reset_btn)
        self._phase_zero_boundary_chk = QCheckBox('Zero at boundary mean')
        self._phase_zero_boundary_chk.setChecked(True)
        self._phase_zero_boundary_chk.setToolTip(
            "Subtract the mean H_k phase of the four boundary-arm edges\n"
            "from every displayed phase value.  Applies to whatever\n"
            "field is currently shown (sim, normalized, measured), so\n"
            "phase maps from different sources become directly\n"
            "comparable — 0° at the boundaries by construction.\n"
            "Additive with the Phase offset (deg) spinbox above.")
        self._phase_zero_boundary_chk.toggled.connect(self._on_bounds_change)
        pr_lay.addWidget(self._phase_zero_boundary_chk)
        pr_lay.addStretch(1)
        fld_lay.addWidget(phase_row)

        # Colour bar toggle — overlays a floating colorbar on top of the
        # napari canvas (right side, vertically centred).  No embedded
        # canvas in the side panel.
        self._cbar_chk = QCheckBox("Show colour bar (overlaid on canvas)")
        self._cbar_chk.setChecked(True)
        self._cbar_chk.toggled.connect(self._on_cbar_toggle)
        fld_lay.addWidget(self._cbar_chk)
        # The overlay QLabel lives in `self._cbar_overlay`; created /
        # replaced by `_refresh_cbar()`.
        self._cbar_overlay = None
        self._cbar_orig_resize = None
        # Tile filter — restricts measured-Q fields to one tile's
        # measurement instead of best-across-tiles.  Only meaningful
        # when tile positions are loaded; hidden otherwise.
        self._tile_filter_combo = None
        if self.tiles:
            tilef_row = QWidget()
            tilef_lay = QHBoxLayout(tilef_row)
            tilef_lay.setContentsMargins(0, 0, 0, 0)
            tilef_lay.addWidget(QLabel("Tile filter:"))
            self._tile_filter_combo = QComboBox()
            self._tile_filter_combo.addItem(
                "All tiles (best measurement)", userData=None)
            for tid in sorted(self.tiles.keys()):
                self._tile_filter_combo.addItem(
                    f"tile {tid}", userData=int(tid))
            self._tile_filter_combo.setToolTip(
                "When set to a specific tile, every measured-Q field\n"
                "shows that tile's measurement only (not the best\n"
                "across tiles).  Edges with no measurement on the\n"
                "selected tile render as 'no data' grey.\n"
                "Doesn't affect Sim fields or geometry.")
            self._tile_filter_combo.currentIndexChanged.connect(
                self._on_tile_filter_change)
            tilef_lay.addWidget(self._tile_filter_combo)
            tilef_lay.addStretch(1)
            fld_lay.addWidget(tilef_row)
        layout.addWidget(fld_group)

        # Percentile filter on total_snr — gates measured-Q fields by
        # network-wide percentile of Var(fit)/Var(resid).  Percentile is
        # computed against the GRAPH-WIDE cached values (best-of-edge),
        # so the threshold is stable regardless of whether a tile
        # filter is on.  0 = no filtering, default.
        from qtpy.QtWidgets import QDoubleSpinBox
        tspct_row = QWidget()
        tspct_lay = QHBoxLayout(tspct_row)
        tspct_lay.setContentsMargins(0, 0, 0, 0)
        tspct_lay.addWidget(QLabel('Min SNR_H1 (dB):'))
        self._tspct_filter_spin = QDoubleSpinBox()
        self._tspct_filter_spin.setRange(-20.0, 60.0)
        self._tspct_filter_spin.setDecimals(1)
        self._tspct_filter_spin.setSingleStep(1.0)
        self._tspct_filter_spin.setValue(-20.0)
        self._tspct_filter_spin.setToolTip(
            "Hide edges whose SNR_H1 (dB) — the canonical per-edge\n"
            "cardiac-fundamental SNR (Q_H1_snr_db) — is below this\n"
            "absolute threshold.  Only affects measured-Q.  Set to the\n"
            "spin's minimum (−20 dB) to disable filtering.")
        self._tspct_filter_spin.valueChanged.connect(
            self._on_total_snr_pct_change)
        tspct_lay.addWidget(self._tspct_filter_spin)
        tspct_lay.addStretch(1)
        fld_lay.addWidget(tspct_row)

        # H2 fundamental SNR percentile filter — same shape as H1 above.
        tspct_h2_row = QWidget()
        tspct_h2_lay = QHBoxLayout(tspct_h2_row)
        tspct_h2_lay.setContentsMargins(0, 0, 0, 0)
        tspct_h2_lay.addWidget(QLabel('Min SNR_H2 (dB):'))
        self._tspct_h2_filter_spin = QDoubleSpinBox()
        self._tspct_h2_filter_spin.setRange(-20.0, 60.0)
        self._tspct_h2_filter_spin.setDecimals(1)
        self._tspct_h2_filter_spin.setSingleStep(1.0)
        self._tspct_h2_filter_spin.setValue(-20.0)
        self._tspct_h2_filter_spin.setToolTip(
            "Hide edges whose SNR_H2 (dB) — the per-edge second-harmonic\n"
            "SNR (Q_H2_snr_db) — is below this absolute threshold.\n"
            "Combines with the H1 gate above via AND.  Only affects\n"
            "measured-Q.  Set to −20 dB to disable.")
        self._tspct_h2_filter_spin.valueChanged.connect(
            self._on_total_snr_h2_pct_change)
        tspct_h2_lay.addWidget(self._tspct_h2_filter_spin)
        tspct_h2_lay.addStretch(1)
        fld_lay.addWidget(tspct_h2_row)

        # Percentile filter on viscous dissipation Φ — gates measured-Q
        # fields by network-wide percentile of `_meas_dissipation` (W).
        # Tooltip shows the actual distribution so the user knows what
        # cut they're applying in absolute terms.  0 = no filtering.
        diss_row = QWidget()
        diss_lay = QHBoxLayout(diss_row)
        diss_lay.setContentsMargins(0, 0, 0, 0)
        diss_lay.addWidget(QLabel('Hide bottom % by dissipation Φ:'))
        self._dissipation_filter_spin = QDoubleSpinBox()
        self._dissipation_filter_spin.setRange(0.0, 99.0)
        self._dissipation_filter_spin.setDecimals(1)
        self._dissipation_filter_spin.setSingleStep(1.0)
        self._dissipation_filter_spin.setValue(0.0)
        # Compute the network-wide dissipation distribution to populate
        # the tooltip range.  Updated once at startup; not re-derived
        # because the field is precomputed and doesn't change post-load.
        _diss_vals = [
            float(dd.get('_meas_dissipation'))
            for _, _, dd in self.G.edges(data=True)
            if dd.get('_meas_dissipation') is not None
                and np.isfinite(dd.get('_meas_dissipation'))
                and float(dd.get('_meas_dissipation')) > 0
        ]
        if _diss_vals:
            arr = np.asarray(_diss_vals)
            self._cached_dissipation_stats = dict(
                n=int(arr.size), min=float(arr.min()),
                p10=float(np.percentile(arr, 10)),
                p50=float(np.percentile(arr, 50)),
                p90=float(np.percentile(arr, 90)),
                max=float(arr.max()),
            )
            _s = self._cached_dissipation_stats
            tooltip = (
                "Hide edges in the bottom X % of viscous dissipation Φ\n"
                "(W; computed as Φ = r·L·⟨Q²⟩ where r = 8μ/(πR⁴) and\n"
                "⟨Q²⟩ = mean(Q_t²) — see methods.md).  Threshold is\n"
                "computed from the GRAPH-WIDE distribution of measured\n"
                "values, so it's stable regardless of Tile filter.\n"
                "Only affects measured-Q fields.  0 = no filtering.\n\n"
                f"Network-wide Φ distribution ({_s['n']} edges):\n"
                f"  min  : {_s['min']:.2e} W\n"
                f"  10%  : {_s['p10']:.2e} W\n"
                f"  median: {_s['p50']:.2e} W\n"
                f"  90%  : {_s['p90']:.2e} W\n"
                f"  max  : {_s['max']:.2e} W")
        else:
            tooltip = ("Hide bottom X % by dissipation Φ.\n"
                       "(No edges have computed dissipation yet.)")
        self._dissipation_filter_spin.setToolTip(tooltip)
        self._dissipation_filter_spin.valueChanged.connect(
            self._on_dissipation_pct_change)
        diss_lay.addWidget(self._dissipation_filter_spin)
        diss_lay.addStretch(1)
        fld_lay.addWidget(diss_row)

        # Tile browser — load any tile's video by picking from a dropdown,
        # independent of edge selection.  Only shown if positions + video
        # paths are configured.
        if bool(self.tiles) and self.video_dir is not None:
            tile_group = QGroupBox("Browse tiles")
            tile_lay = QVBoxLayout(tile_group)
            self._tile_combo = QComboBox()
            # Tile entries: each item carries the int tile ID as
            # userData so the handler can use currentData() directly
            # instead of parsing the label text.  Tiles whose video
            # file isn't present in `video_dir` (typical for
            # partial-coverage bundles like the intern packages) get
            # " (no video)" appended and are dim-disabled.
            self._tile_combo.addItem("(pick a tile)", userData=None)
            n_missing = 0
            for tid in sorted(self.tiles.keys()):
                vid_path = self._get_video_path_for(int(tid))
                label = (f"{tid}"
                          if vid_path is not None
                          else f"{tid}  (no video)")
                self._tile_combo.addItem(label, userData=int(tid))
                if vid_path is None:
                    n_missing += 1
                    idx = self._tile_combo.count() - 1
                    item = self._tile_combo.model().item(idx)
                    if item is not None:
                        from qtpy.QtCore import Qt as _Qt
                        item.setFlags(item.flags()
                                       & ~_Qt.ItemIsEnabled
                                       & ~_Qt.ItemIsSelectable)
            tip = ("Pick a tile ID then click 'Load this tile' to "
                    "overlay\nits raw 250-fps video on the mosaic.")
            if n_missing:
                tip += (f"\n\n{n_missing} tile(s) are greyed out "
                         f"because their video files\nare not present "
                         f"in this bundle.")
            self._tile_combo.setToolTip(tip)
            tile_lay.addWidget(self._tile_combo)
            self._tile_load_btn = QPushButton('Load this tile video')
            self._tile_load_btn.clicked.connect(self._load_video_from_dropdown)
            tile_lay.addWidget(self._tile_load_btn)
            layout.addWidget(tile_group)

        # Display toggles
        d_group = QGroupBox("Display")
        d_lay = QVBoxLayout(d_group)
        self._nodes_chk = QCheckBox("Show nodes")
        self._nodes_chk.toggled.connect(self._on_nodes_toggle)
        d_lay.addWidget(self._nodes_chk)
        if bool(self.tiles):
            self._tile_boundaries_chk = QCheckBox("Show tile boundaries")
            self._tile_boundaries_chk.toggled.connect(
                self._on_tile_boundaries_toggle)
            d_lay.addWidget(self._tile_boundaries_chk)
            self._tile_labels_chk = QCheckBox("Show tile numbers")
            self._tile_labels_chk.toggled.connect(
                self._on_tile_labels_toggle)
            d_lay.addWidget(self._tile_labels_chk)
        # Edge value labels (only render when Tile filter selects a
        # single tile — at All-tiles scale the labels overlap badly).
        self._edge_labels_chk = QCheckBox("Show edge value labels (single tile)")
        self._edge_labels_chk.setToolTip(
            "Show the current Property value at each edge's midpoint.\n"
            "Only renders when Tile filter is set to a specific tile —\n"
            "in All-tiles mode this would clutter the mosaic.\n"
            "Labels update when you change Property / Harmonic / tile.")
        self._edge_labels_chk.toggled.connect(self._on_edge_labels_toggle)
        d_lay.addWidget(self._edge_labels_chk)
        # Kirchhoff KCL residuals: per-node signed sum of incident Q
        # for the selected tile's measurements.  Internal nodes should
        # net to ~0 (mass conservation); large residuals flag measurement
        # / geometry inconsistencies.
        self._kcl_chk = QCheckBox("Show Kirchhoff residuals (tile mode)")
        self._kcl_chk.setToolTip(
            "Per-node signed sum of incident Q on this tile's measurements.\n"
            "For interior nodes, KCL says the sum should be ≈ 0 (mass\n"
            "conservation).  Non-zero residuals at interior nodes are\n"
            "the joint measurement-noise + geometry-misfit + topology\n"
            "error.  Boundary nodes carry the net flow into/out of the\n"
            "carve at the boundary, expected to be non-zero.\n"
            "Only renders when Tile filter selects a specific tile.\n"
            "Diverging colourmap: red = net outflow, blue = net inflow.")
        self._kcl_chk.toggled.connect(self._on_kcl_toggle)
        d_lay.addWidget(self._kcl_chk)
        layout.addWidget(d_group)

        layout.addStretch()

        # ══════ Tab 2: TIME SERIES ════════════════════════════════════
        layout = ts_layout
        # Edge inspection panel (now lives in the time-series tab).
        inspect_group = QGroupBox("Selected edge")
        inspect_lay = QVBoxLayout(inspect_group)
        self._info_box = QTextEdit()
        self._info_box.setReadOnly(True)
        self._info_box.setMaximumHeight(220)
        self._info_box.setPlaceholderText(
            "Click an edge in the viewer to inspect it.")
        inspect_lay.addWidget(self._info_box)
        # ── Go to edge (u, v) — jump-to-vessel by node ids ──────────────
        goto_row = QHBoxLayout()
        goto_row.addWidget(QLabel("Go to (u,v):"))
        self._goto_edge_input = QLineEdit()
        self._goto_edge_input.setPlaceholderText(
            "e.g. 89617, 90142  or  (89617, 90142)")
        self._goto_edge_input.returnPressed.connect(self._goto_edge_from_input)
        goto_row.addWidget(self._goto_edge_input, 1)
        self._goto_edge_btn = QPushButton("Go")
        self._goto_edge_btn.setToolTip(
            "Select the edge with the given node ids (u,v).  Reuses the "
            "normal click-selection path — populates info box, highlights "
            "the vessel, centers the viewport, loads video if auto-load is "
            "on.  Then click 'Compute fresh PIV/OF' + 'Show OF pipeline "
            "walkthrough' to see the diagnostic figure.")
        self._goto_edge_btn.clicked.connect(self._goto_edge_from_input)
        goto_row.addWidget(self._goto_edge_btn)
        inspect_lay.addLayout(goto_row)
        self._plot_qt_btn = QPushButton("Plot Q(t) for selected")
        self._plot_qt_btn.setEnabled(False)
        self._plot_qt_btn.clicked.connect(self._plot_selected_qt)
        inspect_lay.addWidget(self._plot_qt_btn)
        # ── Overlay: compare current source with a second source ────────
        _cmp_row = QHBoxLayout()
        _cmp_row.addWidget(QLabel("Compare current source with:"))
        from qtpy.QtWidgets import QComboBox
        self._overlay_source_combo = QComboBox()
        self._overlay_source_combo.addItem("(none)", None)
        self._overlay_source_combo.addItem("measured", "measured")
        self._overlay_source_combo.addItem("reconciled", "reconciled")
        self._overlay_source_combo.addItem("sim (intact)", "sim")
        self._overlay_source_combo.addItem("sim (carved)", "sim_carved")
        self._overlay_source_combo.setToolTip(
            "If set (≠ (none)), 'Plot Q(t)' overlays this second source's "
            "reconstructed Q(t) on the same axes as the primary source "
            "(the one selected by the View-tab Source radios).\n"
            "Uses stored harmonic phasors: DC + H1/H2/H3.")
        _cmp_row.addWidget(self._overlay_source_combo)
        _cmp_row.addStretch(1)
        self._plot_overlay_btn = QPushButton("Plot overlay")
        self._plot_overlay_btn.setEnabled(False)
        self._plot_overlay_btn.setToolTip(
            "Plot Q(t) for BOTH the current source (View-tab Source radio) "
            "and the 'Compare with' source above, overlaid on one axis.\n"
            "Both waveforms are reconstructed from stored harmonics (DC+H1..H3).")
        self._plot_overlay_btn.clicked.connect(self._plot_selected_qt_overlay)
        _cmp_row.addWidget(self._plot_overlay_btn)
        inspect_lay.addLayout(_cmp_row)
        self._plot_bnd_kcl_btn = QPushButton(
            "Plot KCL sum at boundary (selected edge's boundary node)")
        self._plot_bnd_kcl_btn.setEnabled(False)
        self._plot_bnd_kcl_btn.setToolTip(
            "For a selected boundary EDGE (arm from V/A tip to first "
            "junction J): pop a figure showing the KCL sum of J's OTHER "
            "incident vessels (tile 14 preferred), which by KCL equals "
            "the flow through the arm.  This is the 'true' boundary Q(t) "
            "your colleague should use as a BC — not the noisy arm PIV.\n\n"
            "Enabled only when the selected edge touches a boundary node.")
        self._plot_bnd_kcl_btn.clicked.connect(self._on_plot_boundary_kcl)
        inspect_lay.addWidget(self._plot_bnd_kcl_btn)
        # Run Farneback optical flow/PIV on the selected edge from the
        # tile video.  Read-only: no graph mutation, no measurement
        # persistence — just shows Q(t), spectrum, and a curved ROI
        # overlay.  The implementation detects spatially-downsampled
        # demo videos and adjusts the effective pixel size accordingly.
        self._run_of_btn = QPushButton("Compute fresh PIV/OF for selected")
        self._run_of_btn.setEnabled(False)
        self._run_of_btn.setToolTip(
            "Reload the tile video, run Farneback OF/PIV on the\n"
            "selected vessel, and pop up Q(t) + spectrum.  No data\n"
            "is written back to the graph.  Requires --tile-positions\n"
            "and --video-dir; ~5-10s per call.")
        self._run_of_btn.clicked.connect(self._on_run_optical_flow_clicked)
        inspect_lay.addWidget(self._run_of_btn)
        # Merge the whole degree-2 vessel chain before running OF (default on).
        # Degree-2 nodes are pass-through points where KCL is trivial, so the
        # per-segment subdivision only costs signal; merging runs OF on the
        # full vessel.  Toggle off to compare against the single clicked segment.
        from qtpy.QtWidgets import QCheckBox as _QCheckBox
        self._of_merge_chk = _QCheckBox("Merge degree-2 vessel chain (whole vessel)")
        self._of_merge_chk.setChecked(True)
        self._of_merge_chk.setToolTip(
            "Run OF on the whole vessel (all degree-2 segments merged) instead\n"
            "of just the clicked segment.  More cross-sections, higher SNR, one\n"
            "consistent flow direction.  Off = single-segment (legacy) behaviour.")
        inspect_lay.addWidget(self._of_merge_chk)
        # Toggle OF-driven centerline refinement.  When OFF, the raw graph
        # centerline is used as-is — useful when refinement mis-tracks (e.g.
        # pushes the CL off-axis on curved / bimodal vessels, giving a
        # non-parabolic profile).  Default ON (with the do-no-harm revert guard).
        self._of_refine_chk = _QCheckBox("Refine centerline (OF ridge-snap)")
        self._of_refine_chk.setChecked(True)
        self._of_refine_chk.setToolTip(
            "Snap the centerline onto the OF velocity ridge before analysis.\n"
            "OFF = use the raw graph centerline as-is.  Turn OFF if the refined\n"
            "CL drifts off the vessel (bimodal / non-parabolic radial profile).")
        inspect_lay.addWidget(self._of_refine_chk)
        # Clear the curved OF-region overlay (created by Run OF; persists
        # across mosaic interactions until cleared).  Disabled when no
        # overlay is active.
        self._clear_of_region_btn = QPushButton("Clear OF region overlay")
        self._clear_of_region_btn.setEnabled(False)
        self._clear_of_region_btn.setToolTip(
            "Remove the curved orange ribbon that highlights the last\n"
            "OF integration region on the mosaic.")
        self._clear_of_region_btn.clicked.connect(
            self._tear_down_of_region_overlay)
        inspect_lay.addWidget(self._clear_of_region_btn)
        # Comprehensive stage-by-stage OF pipeline walkthrough for the last
        # fresh OF run.  ('Compute fresh PIV/OF' already pops it automatically;
        # this button re-opens it.)  Folds in what used to be three separate
        # popups — the 4-panel diagnostic, the Kloosterman per-slice figure,
        # and the vessel-quality verdict — into one single figure driven by
        # one data path.
        self._of_pipeline_btn = QPushButton("Show OF pipeline walkthrough")
        self._of_pipeline_btn.setEnabled(False)
        self._of_pipeline_btn.setToolTip(
            "Re-open the full stage-by-stage OF pipeline figure for the last "
            "fresh OF run: ROI selection, background subtraction, mean flow "
            "field, centerline refinement (coarse shift → per-slice snap → "
            "offset smoothing), vessel mask + tangent, radial-profile binning, "
            "the three radii + how each is chosen, the Kloosterman per-slice "
            "gate with reject reasons, Q-estimator comparison, and Q(t) — plus "
            "an overall PASS / CROP / RAILING verdict.")
        self._of_pipeline_btn.clicked.connect(self._show_of_pipeline_figures)
        inspect_lay.addWidget(self._of_pipeline_btn)


        # Comparison set — accumulate edges across multiple clicks and
        # overlay their Q(t)s on a single figure.  Useful for comparing
        # nearby vessels' waveforms.
        self._add_to_comp_btn = QPushButton("Add to comparison set")
        self._add_to_comp_btn.setEnabled(False)
        self._add_to_comp_btn.setToolTip(
            "Add the currently-selected edge to a comparison set.\n"
            "Use 'Plot comparison' to see all of them overlaid.")
        self._add_to_comp_btn.clicked.connect(self._add_to_comparison)
        inspect_lay.addWidget(self._add_to_comp_btn)
        self._plot_comp_btn = QPushButton("Plot comparison (0)")
        self._plot_comp_btn.setEnabled(False)
        self._plot_comp_btn.clicked.connect(self._plot_comparison)
        inspect_lay.addWidget(self._plot_comp_btn)
        self._clear_comp_btn = QPushButton("Clear comparison")
        self._clear_comp_btn.setEnabled(False)
        self._clear_comp_btn.clicked.connect(self._clear_comparison)
        inspect_lay.addWidget(self._clear_comp_btn)
        # Video overlay controls — only shown if positions + video paths
        # were given on the CLI.  Otherwise hide them rather than confuse
        # the user.
        video_enabled = bool(self.tiles) and self.video_dir is not None
        self._auto_video_chk = QCheckBox("Auto-load video on edge click")
        self._auto_video_chk.setEnabled(video_enabled)
        self._auto_video_chk.setToolTip(
            "When ON, clicking an edge automatically loads + overlays\n"
            "the raw video for that tile.  Loading takes ~5-10s per\n"
            "tile; the previous overlay is removed each click.")
        inspect_lay.addWidget(self._auto_video_chk)
        self._video_btn = QPushButton("Load video for selected tile")
        self._video_btn.setEnabled(False)
        self._video_btn.clicked.connect(self._load_video_for_selected)
        inspect_lay.addWidget(self._video_btn)
        self._unload_video_btn = QPushButton("Remove video overlay")
        self._unload_video_btn.setEnabled(False)
        self._unload_video_btn.clicked.connect(
            lambda: (self._remove_video_overlay(),
                     self._unload_video_btn.setEnabled(False)))
        inspect_lay.addWidget(self._unload_video_btn)
        if not video_enabled:
            note = QLabel(
                "<small><i>Video overlay disabled — pass <code>"
                "--tile-positions</code> + <code>--video-dir</code> + "
                "<code>--video-pattern</code> (or <code>--config</code>) "
                "to enable.</i></small>")
            note.setWordWrap(True)
            inspect_lay.addWidget(note)
        layout.addWidget(inspect_group)
        layout.addStretch()

        # ══════ Tab 3: SIMULATE (unified — intact + carved recipes) ═══
        # One tab, source-recipe radio at the top toggles between:
        #   - Intact: 2 sources at DA anchors, α auto-fit on DC
        #   - Carved: N sources at porous outlets, per-zone Σα constrained fit
        # Both share the LOCKED two-pass V recipe and the same D_areal knob.
        from qtpy.QtWidgets import (QRadioButton as _QRB,
                                     QButtonGroup as _QBG,
                                     QGroupBox as _QGB,
                                     QVBoxLayout as _QVL,
                                     QWidget as _QW)
        layout = sim_layout
        _recipe_group = _QGB("Source recipe")
        _recipe_lay = _QVL(_recipe_group)
        self._sim_recipe_intact_radio = _QRB(
            "Intact — 2 sources at DA anchors, α auto-fit on DC")
        self._sim_recipe_carved_radio = _QRB(
            "Carved — N sources at porous outlets "
            "(requires tagged edges in Uncertain tab)")
        self._sim_recipe_intact_radio.setChecked(True)
        _bg = _QBG(_recipe_group)
        _bg.addButton(self._sim_recipe_intact_radio)
        _bg.addButton(self._sim_recipe_carved_radio)
        _recipe_lay.addWidget(self._sim_recipe_intact_radio)
        _recipe_lay.addWidget(self._sim_recipe_carved_radio)
        layout.addWidget(_recipe_group)

        # Intact recipe container — holds the existing simulate-tab widgets.
        self._sim_intact_panel = _QW()
        _intact_container_lay = _QVL(self._sim_intact_panel)
        _intact_container_lay.setContentsMargins(0, 0, 0, 0)
        self._build_simulate_tab(_intact_container_lay)
        _intact_container_lay.addStretch()
        layout.addWidget(self._sim_intact_panel)

        # Carved recipe container — holds the carved-sim widgets.
        self._sim_carved_panel = _QW()
        _carved_container_lay = _QVL(self._sim_carved_panel)
        _carved_container_lay.setContentsMargins(0, 0, 0, 0)
        self._build_carved_sim_tab(_carved_container_lay)
        self._sim_carved_panel.setVisible(False)
        layout.addWidget(self._sim_carved_panel)

        # Gate the carved-recipe radio: only enabled if the graph has
        # tagged edges to define a porous zone.
        _n_tagged = sum(1 for _, _, d in self.G.edges(data=True)
                        if d.get('radial_uncertain'))
        if _n_tagged == 0:
            self._sim_recipe_carved_radio.setEnabled(False)
            self._sim_recipe_carved_radio.setToolTip(
                "No edges tagged `radial_uncertain` — tag uncertain "
                "edges in the Uncertain tab first.")

        def _on_recipe_toggle(_checked=None):
            is_carved = self._sim_recipe_carved_radio.isChecked()
            self._sim_intact_panel.setVisible(not is_carved)
            self._sim_carved_panel.setVisible(is_carved)
        self._sim_recipe_intact_radio.toggled.connect(_on_recipe_toggle)
        self._sim_recipe_carved_radio.toggled.connect(_on_recipe_toggle)
        layout.addStretch()

        # Info label (View tab footer)
        _mode_txt = ("<b>Workbench</b> — fresh OF, tagging and boundary edits; saves go to the stage graph."
                     if self.workbench else
                     "<b>Viewer</b> — read-only: nothing is written, no OF re-runs."
                     + (" (frozen release)" if self.is_release else ""))
        info = QLabel(
            f"<small>{self.G.number_of_edges()} edges, "
            f"{self.G.number_of_nodes()} nodes<br>{_mode_txt}</small>")
        info.setWordWrap(True)
        view_layout.addWidget(info)

        # Assemble the tabbed panel
        tabs.addTab(view_panel, "View")
        tabs.addTab(ts_panel, "Time series")
        # Wrap the Simulate tab in a scroll area so the Run button is always
        # reachable even when optional sections are expanded.
        _sim_scroll = QScrollArea()
        _sim_scroll.setWidgetResizable(True)
        _sim_scroll.setWidget(sim_panel)
        # Simulations are the modelling side (plexusmech); workbench only.
        if self.workbench and self.mech_available:
            tabs.addTab(_sim_scroll, "Simulate")
        # (The Carved Sim tab is now folded into the Simulate tab as the
        # "Carved" source-recipe option — one tab, radio-toggled.)
        # Edit tab: boundary-node marking (source/sink/edge/clear) + save.
        # The rest of the viewer stays read-only.
        edit_panel = self._build_edit_panel()
        _edit_scroll = QScrollArea()
        _edit_scroll.setWidgetResizable(True)
        _edit_scroll.setWidget(edit_panel)
        if self.workbench:
            tabs.addTab(_edit_scroll, "Edit")
        self._edit_mode = None
        self._apply_capabilities()
        self.viewer.window.add_dock_widget(tabs, name="Controls", area='right')
        self._selected_edge: Optional[Tuple[int, int]] = None
        # Radial-uncertainty tagging: viewer-editable per-edge flag
        # (``radial_uncertain=True``) that overrides the edge colour with
        # RADIAL_UNCERTAIN_COLOR when highlighting is on.  Solver
        # behaviour is unchanged — this is a downstream-analysis marker.
        self._show_radial_uncertain = False   # off by default (opt in via checkbox)
        self._radial_tags_dirty = False       # unsaved changes tracker
        # napari keybinding: ``u`` toggles the flag on the currently
        # selected edge.  Registered here (after viewer + selection state
        # exist) so it survives layer rebuilds.
        if self.workbench:
            try:
                @self.viewer.bind_key('u', overwrite=True)
                def _toggle_radial_uncertain_kb(_v):
                    self._toggle_radial_uncertain_on_selected()
            except Exception:
                pass
        # Build the initial colour bar to reflect current field.
        self._refresh_cbar()


    # ── core: refresh edge rendering ───────────────────────────────────
    def _refresh_edges(self):
        # Always draw every edge in the network.  Edges with a usable
        # PIV measurement get coloured by the selected field; edges with
        # no measurement or with measurement-quality flagged garbage
        # (gated, flag-on-delta, fit failed) get a single dim-grey
        # colour so the topology stays visible without pretending we
        # have data we don't.
        field = self.current_field
        is_categorical = (self.current_property == 'harmonic_class'
                          or 'quality_tier' in field)
        # total_snr percentile filter — applies only to measured-Q
        # fields.  Threshold computed ONCE from the network-wide
        # `_h_total_snr` distribution; per-edge gate below.  Re-derive
        # if the percentile changed (handler clears the cache).
        ts_filter_active = (
            float(self.total_snr_pct_filter) > -20.0
            and self.current_source == 'measured'
            and self.current_quantity == 'Q'
        )
        # Dissipation percentile filter — applies only to measured-Q.
        ds_filter_active = (
            float(self.dissipation_pct_filter) > 0.0
            and self.current_source == 'measured'
            and self.current_quantity == 'Q'
        )
        ds_threshold: Optional[float] = None
        if ds_filter_active:
            if self._cached_dissipation_threshold is None:
                _ds_vals = [
                    float(dd.get('_meas_dissipation'))
                    for _, _, dd in self.G.edges(data=True)
                    if dd.get('_meas_dissipation') is not None
                        and np.isfinite(dd.get('_meas_dissipation'))
                        and float(dd.get('_meas_dissipation')) > 0
                ]
                if _ds_vals:
                    self._cached_dissipation_threshold = float(
                        np.percentile(np.asarray(_ds_vals),
                                       self.dissipation_pct_filter))
                else:
                    self._cached_dissipation_threshold = -np.inf
            ds_threshold = self._cached_dissipation_threshold

        ts_threshold: Optional[float] = None
        if ts_filter_active:
            ts_threshold = float(self.total_snr_pct_filter)

        # H2 SNR absolute-dB filter (mirrors H1).
        ts_h2_filter_active = (
            float(self.total_snr_h2_pct_filter) > -20.0
            and self.current_source == 'measured'
            and self.current_quantity == 'Q'
        )
        ts_h2_threshold: Optional[float] = None
        if ts_h2_filter_active:
            ts_h2_threshold = float(self.total_snr_h2_pct_filter)

        def _best_per_tile_snr(edge_d: dict, k_target: int) -> float:
            """Max across measuring tiles of the per-tile harmonic-fit SNR
            (dB) for harmonic k_target.  Returns −inf if none.  Reads
            from measurements_piv because Q_H<k>_snr_db is not always
            precomputed as a top-level edge attr."""
            best = -np.inf
            for m in (edge_d.get('measurements_piv') or []):
                for h in (m.get('harm', {}).get('harmonics') or []):
                    if int(h.get('k', 0)) != k_target:
                        continue
                    s = h.get('snr_db')
                    if s is not None and np.isfinite(s) and float(s) > best:
                        best = float(s)
                    break
            return best

        EDGE_WIDTH_MIN = 0.5
        EDGE_WIDTH_MAX = 22.0
        lines: List[np.ndarray] = []
        vals: List[Optional[float]] = []   # None ⇒ no usable measurement
        tiers: List[Optional[str]] = []
        widths: List[float] = []
        H_mosaic = self.mosaic_height
        for (u, v) in self.edge_list:
            d = self.G.edges[u, v]
            m = _best_measurement(d.get('measurements_piv'))
            x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
            x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            # napari shapes layer wants (row, col) = (display_y, x);
            # display_y = mosaic_height − graph_y.  Straight node-to-node line
            # for every edge (robust: all edges render).  The true curved
            # centerline is drawn only for the selected edge (_highlight_edge_path).
            lines.append(np.array([[H_mosaic - y1, x1],
                                    [H_mosaic - y2, x2]]))
            # Edge width ≈ vessel radius (NOT diameter — keeps neighbour
            # edges from overlapping at large radii while still being
            # visually proportional).
            r_px = float(d.get('radius') or 1.0)
            widths.append(float(np.clip(r_px,
                                         EDGE_WIDTH_MIN, EDGE_WIDTH_MAX)))
            usable = _measurement_usable(m)
            if is_categorical:
                # Two categorical fields, different sources:
                #   quality_tier   — per-measurement (tier letter)
                #   harmonic_class — top-level edge attr (int 0-3)
                # Apply the same total_snr percentile gate so the
                # filter is consistent across continuous + categorical
                # views of measured-Q data.
                if (ts_filter_active and ts_threshold is not None):
                    edge_tsnr = _best_per_tile_snr(d, 1)
                    gate_passes = (np.isfinite(edge_tsnr)
                                    and edge_tsnr >= ts_threshold)
                else:
                    gate_passes = True
                # H2 SNR gate combines with H1/dissipation via AND.
                if (gate_passes and ts_h2_filter_active
                        and ts_h2_threshold is not None):
                    edge_h2 = _best_per_tile_snr(d, 2)
                    if (not np.isfinite(edge_h2)
                            or edge_h2 < ts_h2_threshold):
                        gate_passes = False
                # Dissipation gate combines with SNR gate via AND.
                if (gate_passes and ds_filter_active
                        and ds_threshold is not None):
                    edge_diss = d.get('_meas_dissipation')
                    if (edge_diss is None
                            or not np.isfinite(edge_diss)
                            or float(edge_diss) < ds_threshold):
                        gate_passes = False
                if not gate_passes:
                    tiers.append(None)
                elif self.current_property == 'harmonic_class':
                    tiers.append(d.get('harmonic_class'))
                elif 'quality_tier' in field:
                    tiers.append(m.get('quality_tier') if usable else None)
                else:
                    tiers.append(None)
                vals.append(None)
            else:
                # 4-selector resolver — already handles measured/sim
                # routing and harmonic indexing.  Returns None for
                # invalid combos or missing data.
                val = self._resolve_field_value(u, v)
                # total_snr percentile gate.  Always checks the
                # graph-wide cached `_h_total_snr` (best-of-edge), so
                # the filter is stable across tile-filter changes.
                if (ts_filter_active and val is not None
                        and ts_threshold is not None):
                    edge_tsnr = _best_per_tile_snr(d, 1)
                    if (not np.isfinite(edge_tsnr)
                            or edge_tsnr < ts_threshold):
                        val = None
                # H2 SNR gate combines with H1 via AND.
                if (ts_h2_filter_active and val is not None
                        and ts_h2_threshold is not None):
                    edge_h2 = _best_per_tile_snr(d, 2)
                    if (not np.isfinite(edge_h2)
                            or edge_h2 < ts_h2_threshold):
                        val = None
                # Dissipation gate combines with SNR gate via AND.
                if (ds_filter_active and val is not None
                        and ds_threshold is not None):
                    edge_diss = d.get('_meas_dissipation')
                    if (edge_diss is None
                            or not np.isfinite(edge_diss)
                            or float(edge_diss) < ds_threshold):
                        val = None
                vals.append(val if (val is not None
                                    and np.isfinite(val)) else None)
                tiers.append(None)

        # Compute colors.  Edges with `vals[i] is None` (no usable
        # measurement) get NO_DATA_COLOR regardless of the chosen field.
        if is_categorical:
            cat_map = (HARMONIC_CLASS_COLORS
                       if field == 'harmonic_class' else TIER_COLORS)
            colors = [
                cat_map.get(t, NO_DATA_COLOR) if t is not None
                else NO_DATA_COLOR
                for t in tiers]
        else:
            finite_arr = np.array(
                [v for v in vals if v is not None], dtype=float)
            if len(finite_arr) == 0:
                colors = [NO_DATA_COLOR] * len(lines)
            else:
                # Signed / cyclic fields render on a fixed linear scale —
                # log10 is meaningless for them.
                diverging = ('waveform_av' in field) or ('signed' in field)
                if self.log_scale and not diverging and 'phase' not in field:
                    pos = finite_arr[finite_arr > 0]
                    if len(pos) == 0:
                        finite_eff = finite_arr
                    else:
                        finite_eff = np.log10(pos)
                else:
                    finite_eff = finite_arr
                # Bounds: 'Percentile' clips to the [lo, hi] pct of the
                # finite per-edge distribution (auto-adaptive; defaults
                # 2/98).  'Manual' uses the user's exact vmin/vmax.
                bounds_mode = 'Percentile'
                _bm_combo = getattr(self, '_bounds_mode_combo', None)
                if _bm_combo is not None:
                    bounds_mode = _bm_combo.currentText()
                if bounds_mode == 'Manual':
                    vlo = float(getattr(self, '_vmin_spin', None).value()
                                 if getattr(self, '_vmin_spin', None)
                                 else 0.0)
                    vhi = float(getattr(self, '_vmax_spin', None).value()
                                 if getattr(self, '_vmax_spin', None)
                                 else 1.0)
                    if vlo > vhi:
                        vlo, vhi = vhi, vlo
                else:
                    p_lo = float(getattr(self, '_pct_lo_spin', None).value()
                                 if getattr(self, '_pct_lo_spin', None)
                                 else 2.0)
                    p_hi = float(getattr(self, '_pct_hi_spin', None).value()
                                 if getattr(self, '_pct_hi_spin', None)
                                 else 98.0)
                    if p_lo > p_hi:
                        p_lo, p_hi = p_hi, p_lo
                    vlo, vhi = np.percentile(finite_eff, [p_lo, p_hi])
                if vlo == vhi:
                    vhi = vlo + 1e-9
                if 'phase' in field:
                    cmap = plt.get_cmap('hsv')
                    # π-unified phase display (phase_gradient_recipe.md):
                    # phase = 0 (red) at the boundary reference, hue
                    # rotates through yellow / green / cyan / blue /
                    # magenta with increasing lag, back to red at π.
                    # Values are folded onto [0, π] before mapping so a
                    # π-flipped tile-gauge branch renders the same as
                    # its unflipped counterpart — the (u,v)/gauge sign
                    # is treated as gauge-freedom, not physics.
                    _vlo, _vhi = 0.0, float(np.pi)
                    norm = mcolors.Normalize(vmin=_vlo, vmax=_vhi)
                    self._last_vmin, self._last_vmax = _vlo, _vhi
                    # Optional constant offset (degrees -> radians).
                    self._phase_offset_rad = float(np.radians(
                        getattr(self, '_phase_offset_spin', None).value()
                        if getattr(self, '_phase_offset_spin', None)
                        else 0.0))
                    # Auto-zero at boundary mean: subtract the CIRCULAR
                    # mean of the current field over the four boundary-
                    # arm edges so every source (sim / normalized /
                    # measured) shows 0° at the boundaries → directly
                    # comparable phase maps across sources.
                    self._phase_boundary_offset_rad = 0.0
                    if getattr(self, '_phase_zero_boundary_chk', None) is not None \
                            and self._phase_zero_boundary_chk.isChecked():
                        _b_ph = []
                        for _i, (_u, _v) in enumerate(self.edge_list):
                            _u_bt = self.G.nodes[_u].get('boundary_type')
                            _v_bt = self.G.nodes[_v].get('boundary_type')
                            if _u_bt in ('source', 'sink') or _v_bt in ('source', 'sink'):
                                _vv = vals[_i]
                                if _vv is not None and np.isfinite(_vv):
                                    _b_ph.append(float(_vv))
                        if _b_ph:
                            _cm = np.mean(np.exp(1j * np.asarray(_b_ph)))
                            self._phase_boundary_offset_rad = float(np.angle(_cm))
                elif diverging:
                    # Diverging map centred at 0: blue = venous-like,
                    # red = arterial-like.  Under Percentile mode the
                    # limits are auto-symmetrised; under Manual mode
                    # we respect the user's exact vmin/vmax (so you can
                    # deliberately show an asymmetric divergent map).
                    cmap = plt.get_cmap('coolwarm')
                    if bounds_mode == 'Manual':
                        norm = mcolors.Normalize(vmin=float(vlo), vmax=float(vhi))
                        self._last_vmin, self._last_vmax = float(vlo), float(vhi)
                    else:
                        vmax_sym = float(max(abs(vlo), abs(vhi), 1e-9))
                        norm = mcolors.Normalize(vmin=-vmax_sym, vmax=vmax_sym)
                        self._last_vmin, self._last_vmax = -vmax_sym, vmax_sym
                else:
                    cmap = plt.get_cmap('viridis')
                    norm = mcolors.Normalize(vmin=float(vlo), vmax=float(vhi))
                    self._last_vmin, self._last_vmax = float(vlo), float(vhi)
                colors = []
                _ph_offset = getattr(self, '_phase_offset_rad', 0.0)
                _ph_bnd_offset = getattr(self, '_phase_boundary_offset_rad', 0.0)
                _is_phase = 'phase' in field
                for v_ in vals:
                    if v_ is None:
                        colors.append(NO_DATA_COLOR)
                    else:
                        if _is_phase:
                            # π-unified LAG convention: 0 at boundary
                            # anchors, increases outward with delay.
                            #   lag = (φ_ref − φ_edge) mod 2π
                            #   if lag < π: lag += π   (unify)
                            #   lag -= π               → [0, π)
                            # Sign is opposite of raw φ_edge because
                            # arctan2(-B, A) gives φ that DECREASES with
                            # propagation delay, so lag = ref − φ_edge
                            # is the physically-intuitive direction.
                            rel = (-(float(v_) + _ph_offset)
                                    + _ph_bnd_offset) % (2 * np.pi)
                            if rel < np.pi:
                                rel = rel + np.pi
                            v_use = rel - np.pi
                        elif (self.log_scale and not diverging and v_ > 0):
                            v_use = np.log10(max(v_, 1e-30))
                        else:
                            v_use = v_
                        colors.append(mcolors.to_hex(cmap(norm(v_use))))

        # Radial-uncertainty highlight override.  Runs LAST so it wins
        # over the field-driven colormap.  Only applies when the user
        # has the highlight toggle on.
        if getattr(self, '_show_radial_uncertain', False):
            for i, (u, v) in enumerate(self.edge_list):
                if self.G.edges[u, v].get('radial_uncertain'):
                    colors[i] = RADIAL_UNCERTAIN_COLOR

        # Flow-direction mismatch highlight override (also runs LAST; wins
        # over the radial-uncertain colour when both apply to an edge).
        if getattr(self, '_show_flow_mismatch', False):
            mm = getattr(self, '_flow_mismatch_edges', None)
            if mm is None:
                mm = self._compute_flow_mismatch_edges()
            for i, (u, v) in enumerate(self.edge_list):
                if frozenset((u, v)) in mm:
                    colors[i] = FLOW_MISMATCH_COLOR

        # Uncertain-fit override: colour the near-arterial splitter edges by
        # their fitted conductance correction θ (red = ↑, blue = ↓ vs baseline).
        _unc_chk = getattr(self, '_unc_show_theta', None)
        _unc_theta = getattr(self, '_unc_theta', None)
        if _unc_theta and _unc_chk is not None and _unc_chk.isChecked():
            from matplotlib import cm as _cm
            _tn = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
            for i, (u, v) in enumerate(self.edge_list):
                th = _unc_theta.get(frozenset((u, v)))
                if th is not None and th > 0:
                    colors[i] = mcolors.to_hex(_cm.coolwarm(_tn(np.log10(th))))

        # Rebuild the edges layer from scratch.  Replacing it cleanly
        # avoids a napari Shapes-layer rendering bug where the .data +
        # .edge_color assignments on an existing layer don't always
        # repaint correctly.
        if self._edges_layer is not None:
            try:
                self.viewer.layers.remove(self._edges_layer)
            except (KeyError, ValueError):
                pass
            self._edges_layer = None
        if lines:
            self._edges_layer = self.viewer.add_shapes(
                lines, shape_type='line',
                edge_color=colors,
                edge_width=widths,
                name='Edges', opacity=0.7,
            )
        n_usable = sum(1 for v in vals if v is not None)
        print(f"  Drew {len(lines)} edges  "
              f"({n_usable} coloured by {field}, "
              f"{len(lines) - n_usable} grey: no usable measurement)")
        # Status message via window title
        title = (f"Mosaic (read-only) — {self.graph_path.name}  |  "
                 f"field={field}  |  n_drawn={len(lines)}")
        self.viewer.title = title
        # Refresh the colour-bar overlay to match the new bounds /
        # colormap.  (Old `_cbar_canvas` attribute is gone; the live
        # state is in `_cbar_chk` + `_cbar_overlay`.)
        if getattr(self, '_cbar_chk', None) is not None:
            self._refresh_cbar()
        # Edge value labels (no-op unless toggle on + single tile)
        self._refresh_edge_labels()


    def _refresh_nodes(self):
        if self._nodes_layer is None:
            return
        if not self.show_nodes:
            self._nodes_layer.visible = False
            return
        # Collect per-node positions (with y-flip to align with the mosaic
        # image) and per-node field values (mean of incident edges with
        # usable measurements).  Nodes with no incident usable edge get
        # NO_DATA_COLOR — same convention as the edges layer.
        field = self.current_field
        is_categorical = (self.current_property == 'harmonic_class'
                          or 'quality_tier' in field)
        node_field: Dict[int, list] = {n: [] for n in self.G.nodes()}
        for u, v in self.edge_list:
            d = self.G.edges[u, v]
            m = _best_measurement(d.get('measurements_piv'))
            if is_categorical:
                if not _measurement_usable(m):
                    continue
                if self.current_property == 'harmonic_class':
                    hc = d.get('harmonic_class')
                    if hc is not None:
                        node_field[u].append(int(hc))
                        node_field[v].append(int(hc))
                else:   # quality_tier
                    tier = m.get('quality_tier', 'X')
                    node_field[u].append(QUALITY_RANK.get(tier, 0))
                    node_field[v].append(QUALITY_RANK.get(tier, 0))
            else:
                val = self._resolve_field_value(u, v)
                if val is not None and np.isfinite(val):
                    node_field[u].append(val)
                    node_field[v].append(val)
        # Build points + colors in node-order
        nodes = list(self.G.nodes())
        pts = np.array([[self.mosaic_height - self.G.nodes[n]['y'],
                         self.G.nodes[n]['x']] for n in nodes])
        vals = np.array([np.mean(node_field[n]) if node_field[n] else np.nan
                         for n in nodes], dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0 or is_categorical:
            face = [NO_DATA_COLOR] * len(nodes)
        else:
            arr = vals.copy()
            if self.log_scale:
                pos = finite[finite > 0]
                if len(pos):
                    finite_eff = np.log10(pos)
                    arr = np.where((arr > 0) & np.isfinite(arr),
                                   np.log10(np.maximum(arr, 1e-30)),
                                   np.nan)
                else:
                    finite_eff = finite
            else:
                finite_eff = finite
            vlo, vhi = np.percentile(finite_eff, [2, 98])
            if vlo == vhi:
                vhi = vlo + 1e-9
            cmap = (plt.get_cmap('hsv') if 'phase' in field
                    else plt.get_cmap('viridis'))
            if 'phase' in field:
                # π-unified: fold onto [0, π] (see edge phase pipeline)
                norm = mcolors.Normalize(vmin=0.0, vmax=float(np.pi))
                _ph_offset_node = float(np.radians(
                    getattr(self, '_phase_offset_spin', None).value()
                    if getattr(self, '_phase_offset_spin', None)
                    else 0.0))
                # Auto-zero at boundary mean (per-source; matches edges)
                _ph_bnd_offset_node = 0.0
                if getattr(self, '_phase_zero_boundary_chk', None) is not None \
                        and self._phase_zero_boundary_chk.isChecked():
                    _b_ph = []
                    for _n, _av in zip(nodes, arr):
                        if not np.isfinite(_av): continue
                        if self.G.nodes[_n].get('boundary_type') in ('source','sink'):
                            _b_ph.append(float(_av))
                    if _b_ph:
                        _cm = np.mean(np.exp(1j * np.asarray(_b_ph)))
                        _ph_bnd_offset_node = float(np.angle(_cm))
            else:
                norm = mcolors.Normalize(vmin=float(vlo), vmax=float(vhi))
                _ph_offset_node = 0.0
                _ph_bnd_offset_node = 0.0
            face = []
            _is_phase = 'phase' in field
            for v_ in arr:
                if not np.isfinite(v_):
                    face.append(NO_DATA_COLOR)
                else:
                    if _is_phase:
                        # π-unified LAG convention (matches edge pipeline):
                        # lag = (ref − v) mod 2π, folded to [0, π).
                        rel = (-(float(v_) + _ph_offset_node)
                                + _ph_bnd_offset_node) % (2 * np.pi)
                        if rel < np.pi:
                            rel = rel + np.pi
                        v_use = rel - np.pi
                    else:
                        v_use = v_
                    face.append(mcolors.to_hex(cmap(norm(v_use))))
        self._nodes_layer.data = pts
        if len(pts):
            self._nodes_layer.face_color = face
        self._nodes_layer.visible = True


    def _on_bounds_change(self, _val=None):
        # Any bounds spinbox changed → re-render with the new clip range.
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()


    def _on_bounds_mode_change(self, _idx=None):
        # Toggle visibility of Percentile vs Manual widgets to reduce
        # clutter, and re-render with the newly active bounds source.
        combo = getattr(self, '_bounds_mode_combo', None)
        if combo is None:
            return
        is_manual = (combo.currentText() == 'Manual')
        for w in (getattr(self, '_pct_lo_lbl', None),
                   getattr(self, '_pct_lo_spin', None),
                   getattr(self, '_pct_hi_lbl', None),
                   getattr(self, '_pct_hi_spin', None)):
            if w is not None:
                w.setVisible(not is_manual)
        for w in (getattr(self, '_vmin_lbl', None),
                   getattr(self, '_vmin_spin', None),
                   getattr(self, '_vmax_lbl', None),
                   getattr(self, '_vmax_spin', None)):
            if w is not None:
                w.setVisible(is_manual)
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()


    def _on_total_snr_pct_change(self, val: float):
        """Update the network-wide total_snr percentile filter.  The
        threshold itself is re-derived from cached `_h_total_snr`
        values on every refresh — set independently of any tile filter
        so it stays stable when you swap tiles."""
        self.total_snr_pct_filter = float(val)
        # Force re-derivation of the cached threshold next refresh.
        self._cached_total_snr_threshold = None
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()


    def _on_total_snr_h2_pct_change(self, val: float):
        """Same as `_on_total_snr_pct_change` but for Q_H2_snr_db."""
        self.total_snr_h2_pct_filter = float(val)
        self._cached_total_snr_h2_threshold = None
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()


    def _on_dissipation_pct_change(self, val: float):
        """Update the network-wide dissipation percentile filter.
        Threshold is re-derived from `_meas_dissipation` (W) values
        across the whole graph — stable regardless of Tile filter."""
        self.dissipation_pct_filter = float(val)
        self._cached_dissipation_threshold = None
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()


    def _on_tile_filter_change(self, _idx=None):
        """Switch between best-measurement coloring and per-tile-only
        coloring.  When a tile is selected, repopulates
        `_tile_harmonic_cache` for that tile (~5 ms × edges with a
        measurement on this tile) so the resolver reads per-tile values
        immediately.  Always re-populating keeps the cache consistent
        even if the video-load path wrote a different tile's data
        since the last filter change.
        """
        if self._tile_filter_combo is None:
            return
        new_filter = self._tile_filter_combo.currentData()
        new_filter = int(new_filter) if new_filter is not None else None
        if new_filter == self.current_tile_filter:
            return
        self.current_tile_filter = new_filter
        if new_filter is not None:
            self._populate_tile_harmonic_cache(new_filter)
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()
        if self.show_kcl:
            self._refresh_kcl_overlay()


    def _on_cbar_toggle(self, checked: bool):
        if not bool(checked) and self._cbar_overlay is not None:
            self._cbar_overlay.setParent(None)
            self._cbar_overlay.deleteLater()
            self._cbar_overlay = None
        else:
            self._refresh_cbar()


    def _refresh_cbar(self):
        """Render the colormap legend as a floating QLabel/QPixmap on the
        napari canvas (right side, vertically centred).  Same approach
        as the editing viewer's `_update_colorbar_layer`."""
        # Always tear down the previous overlay (if any) before drawing.
        old = self._cbar_overlay
        if old is not None:
            old.setParent(None)
            old.deleteLater()
            self._cbar_overlay = None
        if not getattr(self, '_cbar_chk', None) \
                or not self._cbar_chk.isChecked():
            return
        from matplotlib.colors import Normalize, LogNorm
        from qtpy.QtWidgets import QLabel
        from qtpy.QtGui import QPixmap, QImage
        from qtpy.QtCore import Qt
        field = self.current_field
        is_categorical = (self.current_property == 'harmonic_class'
                          or 'quality_tier' in field)
        vmin = getattr(self, '_last_vmin', None)
        vmax = getattr(self, '_last_vmax', None)
        if is_categorical or vmin is None or vmax is None:
            return  # nothing meaningful to draw
        is_phase = (self.current_property == 'phase')
        use_log = bool(self.log_scale) and not is_phase
        # Compose colorbar label from the selectors so it always reads
        # cleanly: "Magnitude |·|  (Q, H₁)", "Pulsatility index  (P)", …
        label = self._current_field_label()
        # Build the matplotlib figure (Agg, transparent bg, white text).
        dpi = 100
        if is_phase:
            # Half-circle color wheel for π-unified phase (matches
            # the [0, π] fold applied in the display pipeline).
            fig_sz = 2.8
            fig = plt.figure(figsize=(fig_sz, fig_sz), dpi=dpi)
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection='polar')
            n = 512
            theta = np.linspace(0, np.pi, n + 1)
            r_in, r_out = 0.45, 1.0
            C = np.tile(np.linspace(0, np.pi, n), (1, 1))
            ax.pcolormesh(theta, [r_in, r_out], C,
                           cmap=plt.get_cmap('hsv'),
                           norm=Normalize(vmin=0, vmax=np.pi),
                           shading='auto')
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(1)
            ax.set_thetamin(0); ax.set_thetamax(180)
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False)
            ax.spines['polar'].set_visible(False)
            ax.text(0, 0, label, ha='center', va='center',
                    color='white', fontsize=11, transform=ax.transData)
            fig.patch.set_facecolor((0, 0, 0, 0))
            ax.patch.set_alpha(0)
        else:
            fig_h_in, fig_w_in = 5.0, 1.8
            fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
            cmap = plt.get_cmap('viridis')
            if use_log and vmin > 0:
                norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=ax)
            cbar.set_label(label, color='white', fontsize=14, labelpad=10)
            cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white',
                                           labelsize=11, width=2, length=5)
            fig.patch.set_facecolor((0, 0, 0, 0))
            ax.patch.set_alpha(0)
            fig.subplots_adjust(left=0.05, right=0.35, top=0.97, bottom=0.03)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)
        h, w, _ = buf.shape
        qimg = QImage(buf.data, w, h, 4 * w, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        # Parent to the vispy canvas widget so it floats over the image.
        qt_viewer = self.viewer.window._qt_viewer
        canvas_widget = qt_viewer.canvas.native
        overlay = QLabel(canvas_widget)
        overlay.setPixmap(pixmap)
        overlay.setFixedSize(w, h)
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        overlay.setStyleSheet("background: transparent;")
        overlay.raise_()
        # Position: right edge, vertically centred.
        def _reposition():
            pw = canvas_widget.width()
            ph = canvas_widget.height()
            x = max(0, pw - w - 20)
            y = max(0, (ph - h) // 2)
            overlay.move(x, y)
        _reposition()
        overlay.show()
        # Re-position on canvas resize (preserve any prior handler).
        if self._cbar_orig_resize is None:
            self._cbar_orig_resize = canvas_widget.resizeEvent
        def _on_resize(event, _orig=self._cbar_orig_resize):
            _orig(event)
            cur = self._cbar_overlay
            if cur is not None and cur.isVisible():
                _reposition()
        canvas_widget.resizeEvent = _on_resize
        self._cbar_overlay = overlay


    # ── tile boundaries + labels (optional overlay) ────────────────────
    def _refresh_tile_boundaries(self):
        # Remove existing layer (None on first call) and re-add if toggled on.
        if self._tile_boundaries_layer is not None:
            try:
                self.viewer.layers.remove(self._tile_boundaries_layer)
            except (KeyError, ValueError):
                pass
            self._tile_boundaries_layer = None
        if not self.show_tile_boundaries or not self.tiles:
            return
        rects: List[np.ndarray] = []
        # Raw tile corners in (y, x) = (row, col) order.
        raw_corners = np.array([
            [0.0,                 0.0],
            [0.0,                 TILE_RAW_WIDTH],
            [TILE_RAW_HEIGHT,     TILE_RAW_WIDTH],
            [TILE_RAW_HEIGHT,     0.0],
        ])
        for tid in sorted(self.tiles.keys()):
            t = self.tiles[tid]
            A = t.get('affine')
            if A is None:
                r0 = t['translate_y'] - self._tile_offset_y
                c0 = t['translate_x'] - self._tile_offset_x
                r1 = r0 + TILE_RAW_HEIGHT * t['scale_y']
                c1 = c0 + TILE_RAW_WIDTH * t['scale_x']
                rects.append(np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]]))
                continue
            M = A[:2, :2]
            tvec = A[:2, 2] - np.array([self._tile_offset_y, self._tile_offset_x])
            # (4,2) corners after rotation+scale, plus offset-shifted translate.
            mapped = raw_corners @ M.T + tvec
            rects.append(mapped)
        self._tile_boundaries_layer = self.viewer.add_shapes(
            rects, shape_type='polygon',
            edge_color='yellow', edge_width=2.0,
            face_color='transparent',
            name='Tile boundaries', opacity=0.7,
        )


    def _refresh_tile_labels(self):
        if self._tile_labels_layer is not None:
            try:
                self.viewer.layers.remove(self._tile_labels_layer)
            except (KeyError, ValueError):
                pass
            self._tile_labels_layer = None
        if not self.show_tile_labels or not self.tiles:
            return
        pts: List[List[float]] = []
        labels: List[str] = []
        centre_raw = np.array([TILE_RAW_HEIGHT * 0.5, TILE_RAW_WIDTH * 0.5])
        for tid in sorted(self.tiles.keys()):
            t = self.tiles[tid]
            A = t.get('affine')
            if A is None:
                r0 = t['translate_y'] - self._tile_offset_y
                c0 = t['translate_x'] - self._tile_offset_x
                r_mid = r0 + 0.5 * TILE_RAW_HEIGHT * t['scale_y']
                c_mid = c0 + 0.5 * TILE_RAW_WIDTH * t['scale_x']
            else:
                M = A[:2, :2]
                tvec = (A[:2, 2]
                        - np.array([self._tile_offset_y, self._tile_offset_x]))
                r_mid, c_mid = M @ centre_raw + tvec
            pts.append([r_mid, c_mid])
            labels.append(str(tid))
        self._tile_labels_layer = self.viewer.add_points(
            np.array(pts), size=0, name='Tile numbers',
            text={'string': labels, 'size': 14,
                   'color': 'yellow', 'anchor': 'center'},
        )


    # ── click handler ──────────────────────────────────────────────────
    def _inspect_nearest_boundary_node(
            self, x: float, y: float, max_px: float = 18.0) -> bool:
        """If the click is within `max_px` of a boundary node, display
        its BC harmonics in the info panel and return True.  Otherwise
        return False so the caller can fall through to edge hit-test.

        `x`, `y` are in graph coordinates (x = G.nodes[n]['x'], y is
        the click-row converted to graph y via mosaic_height − row).
        """
        if not self._boundary_nodes:
            return False
        y_graph = self.mosaic_height - y
        best_n, best_d = None, np.inf
        for n in self._boundary_nodes:
            dx = self.G.nodes[n]['x'] - x
            dy = self.G.nodes[n]['y'] - y_graph
            d = float(np.hypot(dx, dy))
            if d < best_d:
                best_n, best_d = n, d
        if best_n is None or best_d > max_px:
            return False
        # Disable edge-only buttons (BC click clears edge selection).
        self._selected_edge = None
        try:
            self._plot_qt_btn.setEnabled(False)
            self._add_to_comp_btn.setEnabled(False)
            self._video_btn.setEnabled(False)
            self._run_of_btn.setEnabled(False)
        except AttributeError:
            pass
        d = self.G.nodes[best_n]
        btype = d.get('boundary_type', '?')
        letter = 'A' if btype == 'source' else 'V' if btype == 'sink' else '?'
        side = ('arterial source' if btype == 'source'
                else 'venous sink' if btype == 'sink' else btype)
        info: List[str] = []
        info.append(f"Boundary node {best_n}  [{letter}]  ({side})")
        info.append(f"  position: ({d.get('x', float('nan')):.0f}, "
                    f"{d.get('y', float('nan')):.0f})")
        bch = d.get('bc_harmonics')
        if bch is None:
            info.append("\n  No bc_harmonics stored on this node.")
        else:
            bch = np.asarray(bch)
            # Apply canonical sign convention so phases at sources and
            # sinks report the same impedance phase, not differ by π:
            #   stored Q at sources = +inflow
            #   stored Q at sinks   = +outflow (network sees −Q)
            # → multiply by −1 at sinks to get Q_signed (always "into
            #   the network"), then read magnitude / phase off that.
            sign = -1.0 if btype == 'sink' else 1.0
            bch_signed = bch * sign
            sign_note = " (Q_signed = −Q_outflow)" if sign < 0 else ""
            info.append(f"\n  bc_harmonics (length {len(bch)}): "
                        f"DC + H₁ + H₂ + H₃{sign_note}")
            labels = ['DC', 'H₁', 'H₂', 'H₃']
            for i, c in enumerate(bch_signed[:4]):
                lab = labels[i] if i < len(labels) else f'H{i}'
                if i == 0:
                    info.append(f"    {lab}:  Q̄ = {float(np.real(c)):+.4g}  "
                                f"(imag = {float(np.imag(c)):+.3g})")
                else:
                    mag = float(np.abs(c))
                    ph = float(np.angle(c))
                    info.append(f"    {lab}:  |a| = {mag:.4g},  "
                                f"∠ = {np.degrees(ph):+.1f}°")
        self._info_box.setPlainText("\n".join(info))
        # Move the click marker to the boundary node so the user sees
        # which one they hit.
        self._click_marker_layer.data = np.array(
            [[self.mosaic_height - d['y'], d['x']]])
        return True


    def _highlight_edge_path(self, u, v):
        """Overlay the selected edge's true curved centerline (its `path`), or a
        straight node-to-node line if the edge has no path.  The main Edges
        layer is drawn straight for rendering robustness; this shows the real
        vessel geometry only for the clicked edge."""
        H = self.mosaic_height
        d = self.G.edges[u, v]
        p = d.get('path')
        if p is not None and len(p) >= 2:
            p = np.asarray(p, dtype=float)
            coords = np.column_stack([H - p[:, 1], p[:, 0]])
        else:
            x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
            x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            coords = np.array([[H - y1, x1], [H - y2, x2]])
        lyr = getattr(self, '_edge_path_layer', None)
        if lyr is not None:
            try:
                self.viewer.layers.remove(lyr)
            except (KeyError, ValueError):
                pass
            self._edge_path_layer = None
        self._edge_path_layer = self.viewer.add_shapes(
            [coords], shape_type='path', edge_color='#00e5ff',
            edge_width=2.5, name='Selected vessel path', opacity=1.0)


    def _inspect_nearest_edge(self, x: float, y: float, max_px: float = 25.0):
        # `y` here is the display row from napari's click event.  The
        # graph stores y in math-convention (image displayed with
        # mosaic_height − y).  Convert click-row back to graph-y so we
        # can compare against precomputed midpoints.
        if len(self._edge_midpoints) == 0:
            return
        y_graph = self.mosaic_height - y
        d2 = ((self._edge_midpoints[:, 0] - x) ** 2 +
              (self._edge_midpoints[:, 1] - y_graph) ** 2)
        i = int(np.argmin(d2))
        dist = float(np.sqrt(d2[i]))
        if dist > max_px:
            self._info_box.setPlainText(
                f"No edge within {max_px} px of click "
                f"(closest was {dist:.1f} px away).")
            self._plot_qt_btn.setEnabled(False)
            self._click_marker_layer.data = np.empty((0, 2))
            return
        u, v = self.edge_list[i]
        self._selected_edge = (u, v)
        self._plot_qt_btn.setEnabled(True)
        if hasattr(self, '_plot_overlay_btn'):
            self._plot_overlay_btn.setEnabled(True)
        # Enable boundary-KCL button only when this edge touches a boundary
        if hasattr(self, '_plot_bnd_kcl_btn'):
            bt_u = self.G.nodes[u].get('boundary_type')
            bt_v = self.G.nodes[v].get('boundary_type')
            self._plot_bnd_kcl_btn.setEnabled(
                bt_u in ('source', 'sink') or bt_v in ('source', 'sink'))
        self._add_to_comp_btn.setEnabled(True)
        # OF button requires a tile + video; gate it on having both.
        _of_ok = bool(self.tiles) and self.of_video_dir is not None
        self._run_of_btn.setEnabled(_of_ok)
        # Place click marker at (mosaic_height − midpoint_y, midpoint_x).
        self._click_marker_layer.data = np.array(
            [[self.mosaic_height - self._edge_midpoints[i, 1],
              self._edge_midpoints[i, 0]]])
        # Overlay the selected edge's TRUE curved centerline (main edges are
        # drawn straight for robustness; the real path shows only on click).
        self._highlight_edge_path(u, v)
        # Enable / drive the video buttons
        m_best = _best_measurement(
            self.G.edges[u, v].get('measurements_piv'))
        # v4 migration: was m_best.get('tile_id')
        self._selected_tile_id = (MeasurementView(m_best).tile_id
                                  if m_best is not None else None)
        can_video = (self._selected_tile_id is not None
                     and bool(self.tiles)
                     and self.video_dir is not None)
        self._video_btn.setEnabled(can_video)
        if can_video and getattr(self, '_auto_video_chk', None) \
                and self._auto_video_chk.isChecked():
            self._load_video_overlay(int(self._selected_tile_id))
            self._unload_video_btn.setEnabled(True)
        # Build readable summary — show ALL measurements for this edge,
        # ordered best-quality-first.  Each measurement has a small
        # marker indicating whether PIV's internal gates accepted it
        # ("✓" usable / "✗" rejected) but we don't surface the tier
        # vocabulary anywhere visible.
        d = self.G.edges[u, v]
        meas = d.get('measurements_piv') or []
        meas_ranked = sorted(
            meas,
            # v4 migration: mm.get('snr_f0') → MeasurementView(mm).snr.h1_db.
            # quality_tier and snr_pulse are not in the mapping table; keep.
            key=lambda mm: (QUALITY_RANK.get(mm.get('quality_tier', 'X'), 0),
                            (MeasurementView(mm).snr.h1_db or 0),
                            mm.get('snr_pulse', 0) or 0),
            reverse=True)
        info: List[str] = []
        info.append(f"Edge ({u}, {v})")
        info.append(f"  endpoints: ({self.G.nodes[u]['x']:.0f}, "
                    f"{self.G.nodes[u]['y']:.0f}) → "
                    f"({self.G.nodes[v]['x']:.0f}, "
                    f"{self.G.nodes[v]['y']:.0f})")
        info.append(f"  radius: {d.get('radius', float('nan')):.1f} px")
        info.append(f"  length: {d.get('length', float('nan')):.1f} px")
        info.append(f"  measurements: {len(meas)} "
                    f"(ranked by quality, best first)")
        display_keys = ('mean_Q', 'amp_Q', 'PI', 'phase', 'snr_pulse',
                        'snr_f0', 'f0_hz', 'v_max')
        # TODO(v4-migration): display_keys still uses legacy flat names
        # ('mean_Q', 'snr_f0', 'f0_hz', ...).  On v4 these won't be present
        # as flat keys on `mm` — need to route each key through MeasurementView
        # (mean_Q → harm.mean_Q, snr_f0 → snr.h1_db, f0_hz → harm.f0_hz,
        #  amp_Q → amp_Q property, PI → PI_rms property, phase → phase_deg,
        #  snr_pulse / v_max not exposed).
        for idx, mm in enumerate(meas_ranked):
            mark = '✓' if _measurement_usable(mm) else '✗'
            # v4 migration: was mm.get('tile_id')
            tile_id = MeasurementView(mm).tile_id
            info.append(f"\n  {mark} measurement {idx + 1} "
                        f"(tile {tile_id}):")
            for k in display_keys:
                if k in mm:
                    val = mm[k]
                    try:
                        info.append(f"    {k}: {float(val):.4g}")
                    except (TypeError, ValueError):
                        info.append(f"    {k}: {val}")
        # Per-harmonic Z-statistics + classification for the best
        # measurement.  Computed on demand (one harmonic fit per click,
        # cheap).  Shows the user which harmonics rise above the noise.
        best = meas_ranked[0] if meas_ranked else None
        best_mv = MeasurementView(best) if best is not None else None
        # v4 migration: was best.get('Q_t') / best.get('f0_hz', 2.5)
        if best is not None and best_mv.Q_t is not None:
            snrs = _harmonic_snrs(
                np.asarray(best_mv.Q_t, dtype=float),
                f0=float(best_mv.harm.f0_hz or 2.5),
                dt=1.0/250)
            if snrs is not None:
                klass = _harmonic_class(snrs)
                info.append("\n  Per-harmonic SNR (Z = amp / SE_amp):")
                for k in ('DC', 'H1', 'H2', 'H3'):
                    pass_mark = '✓' if snrs[k] >= HARMONIC_SNR_THRESHOLD else '·'
                    info.append(f"    {pass_mark} Z({k}) = {snrs[k]:.1f}")
                info.append(f"  Resolved: {_harmonic_class_label(klass)} "
                            f"(threshold Z > {HARMONIC_SNR_THRESHOLD:.0f})")
        self._info_box.setPlainText("\n".join(info))


    def _goto_edge_from_input(self):
        """Jump to an edge by its (u, v) node ids typed in the goto box.

        Reuses `_inspect_nearest_edge` at the edge's midpoint so all the
        normal click-selection state gets populated identically (info
        box, marker, path highlight, tile pick, video auto-load).  Also
        centers the napari viewport on the vessel so it is visible.
        """
        text = (self._goto_edge_input.text() or "").strip()
        if not text:
            return
        cleaned = (text.replace('(', ' ').replace(')', ' ')
                       .replace(',', ' ').replace(';', ' '))
        parts = cleaned.split()
        if len(parts) != 2:
            self._info_box.setPlainText(
                f"Bad Go-to input {text!r}. Expected two node ids like "
                f"'89617, 90142'.")
            return
        try:
            u, v = int(parts[0]), int(parts[1])
        except ValueError:
            self._info_box.setPlainText(
                f"Bad Go-to input {text!r}. Node ids must be integers.")
            return
        # Locate in edge_list (either direction).
        i = None
        for j, (a, b) in enumerate(self.edge_list):
            if {a, b} == {u, v}:
                i = j
                break
        if i is None:
            self._info_box.setPlainText(
                f"Edge ({u}, {v}) not in current graph.")
            return
        xm = float(self._edge_midpoints[i, 0])
        ym_graph = float(self._edge_midpoints[i, 1])
        y_display = float(self.mosaic_height - ym_graph)
        # Route through the click-selection path (max_px=1 forces the same
        # index we already located; distance is zero at the midpoint).
        self._inspect_nearest_edge(xm, y_display, max_px=1.0)
        # Center the napari camera on the vessel without changing zoom.
        try:
            cur_zoom = self.viewer.camera.zoom
            self.viewer.camera.center = (0.0, y_display, xm)
            self.viewer.camera.zoom = cur_zoom
        except Exception:
            pass


    # ── callbacks ──────────────────────────────────────────────────────
    def _sync_field_combos(self):
        """Sync enable / disable state of the Quantity, Property, and
        Harmonic combos to the current Source + selection.  Greys out
        invalid combinations (e.g. P in Measured, Pressure drop in
        Q-mode, Phase at DC) so the user can never pick a bad combo.

        If the current selection becomes invalid after a Source change,
        falls back to the first valid Property / Quantity it finds.
        """
        if not hasattr(self, '_quantity_combo'):
            return
        from qtpy.QtCore import Qt
        # Quantity: P only enabled when Sim source is active.
        sim_active = bool(getattr(self, '_sim_active', False))
        qmodel = self._quantity_combo.model()
        q_p_item = qmodel.item(1)  # 'P' entry
        if q_p_item is not None:
            flags = q_p_item.flags()
            if sim_active and self.current_source == 'sim':
                q_p_item.setFlags(flags | Qt.ItemIsEnabled
                                    | Qt.ItemIsSelectable)
            else:
                q_p_item.setFlags(flags & ~Qt.ItemIsEnabled
                                    & ~Qt.ItemIsSelectable)
                if self.current_quantity == 'P':
                    self.current_quantity = 'Q'
                    self._quantity_combo.blockSignals(True)
                    self._quantity_combo.setCurrentIndex(0)
                    self._quantity_combo.blockSignals(False)
        # Property: enable rows matching (source, quantity, harmonic)
        # validity — delegate to _combo_valid so DC-incompatible
        # properties (phase, PI) grey out symmetrically.
        pmodel = self._property_combo.model()
        first_valid_idx = -1
        for i, (key, _lbl, _h, _valid_in) in enumerate(PROPERTY_DEFS):
            item = pmodel.item(i)
            if item is None:
                continue
            ok = _combo_valid(self.current_source, self.current_quantity,
                               key, self.current_harmonic)
            flags = item.flags()
            if ok:
                item.setFlags(flags | Qt.ItemIsEnabled
                                | Qt.ItemIsSelectable)
                if first_valid_idx < 0:
                    first_valid_idx = i
            else:
                item.setFlags(flags & ~Qt.ItemIsEnabled
                                & ~Qt.ItemIsSelectable)
        # If the currently-selected Property is invalid, fall back.
        if not _combo_valid(self.current_source, self.current_quantity,
                             self.current_property, self.current_harmonic):
            if first_valid_idx >= 0:
                self.current_property = PROPERTY_DEFS[first_valid_idx][0]
                self._property_combo.blockSignals(True)
                self._property_combo.setCurrentIndex(first_valid_idx)
                self._property_combo.blockSignals(False)
        # Harmonic combo: enabled only for harmonic-keyed properties.
        pdef = next((p for p in PROPERTY_DEFS
                      if p[0] == self.current_property), None)
        harmonic_keyed = bool(pdef and pdef[2])
        self._harmonic_combo.setEnabled(harmonic_keyed)
        # Disable DC entry when Property = phase (undefined).
        hmodel = self._harmonic_combo.model()
        dc_item = hmodel.item(0)
        if dc_item is not None:
            flags = dc_item.flags()
            if self.current_property == 'phase':
                dc_item.setFlags(flags & ~Qt.ItemIsEnabled
                                    & ~Qt.ItemIsSelectable)
                if self.current_harmonic == 'DC':
                    self.current_harmonic = 'H1'
                    self._harmonic_combo.blockSignals(True)
                    self._harmonic_combo.setCurrentIndex(1)
                    self._harmonic_combo.blockSignals(False)
            else:
                dc_item.setFlags(flags | Qt.ItemIsEnabled
                                    | Qt.ItemIsSelectable)
        # Reflect Source/Quantity in the time-series "Plot" button label so
        # it's clear whether it shows measured PIV or the simulated waveform.
        if getattr(self, '_plot_qt_btn', None) is not None:
            if self.current_source == 'sim':
                sym = 'P' if self.current_quantity == 'P' else 'Q'
                self._plot_qt_btn.setText(
                    f"Plot simulated {sym}(t) for selected")
            else:
                self._plot_qt_btn.setText("Plot Q(t) for selected")


    def _on_quantity_change(self, _idx=None):
        new_q = self._quantity_combo.currentData()
        if new_q is None or new_q == self.current_quantity:
            return
        self.current_quantity = new_q
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()


    def _on_property_change(self, _idx=None):
        new_p = self._property_combo.currentData()
        if new_p is None or new_p == self.current_property:
            return
        self.current_property = new_p
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()


    def _on_harmonic_change(self, _idx=None):
        new_h = self._harmonic_combo.currentData()
        if new_h is None or new_h == self.current_harmonic:
            return
        self.current_harmonic = new_h
        # Harmonic gates property availability (phase/PI invalid at DC).
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()


    def _on_log_toggle(self, checked: bool):
        self.log_scale = bool(checked)
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()


    # ── Radial-uncertainty tagging ────────────────────────────────

    def _on_radial_uncertain_toggle(self, checked: bool):
        self._show_radial_uncertain = bool(checked)
        self._refresh_edges()


    def _compute_flow_mismatch_edges(self):
        """Find edges measured in ≥2 tiles whose flow DIRECTION disagrees.

        Only measurements with ``snr_harm_fit_db`` ≥ the SNR threshold count
        (below the threshold the phasor is noise-dominated and its sign is
        unidentifiable, so a flip isn't a real disagreement).  An undirected
        edge has only two possible directions, so ≥2 distinct
        ``(flow_from, flow_to)`` pairs among qualifying measurements means the
        tiles disagree.  Caches the result (a set of ``frozenset({u, v})``)
        and refreshes the status label.  Returns the set.
        """
        thr = (float(self._flow_mismatch_snr_spin.value())
               if hasattr(self, '_flow_mismatch_snr_spin') else 3.0)
        mismatched = set()
        n_shared = 0
        for u, v, d in self.G.edges(data=True):
            dirs = set()
            n_qual = 0
            for m in (d.get('measurements_piv') or []):
                mv_m = MeasurementView(m)
                # v4 migration: was m.get('flow_from') / m.get('flow_to')
                ff, ft = mv_m.flow_from, mv_m.flow_to
                if ff is None or ft is None:
                    continue
                # v4 migration: was m.get('snr_harm_fit_db') → snr.dc_db
                snr = mv_m.snr.dc_db
                try:
                    if snr is None or not (float(snr) >= thr):
                        continue
                except (TypeError, ValueError):
                    continue
                n_qual += 1
                dirs.add((ff, ft))
            if n_qual >= 2:
                n_shared += 1
                if len(dirs) >= 2:
                    mismatched.add(frozenset((u, v)))
        self._flow_mismatch_edges = mismatched
        if hasattr(self, '_flow_mismatch_status_lbl'):
            self._flow_mismatch_status_lbl.setText(
                f"Mismatched: {len(mismatched)} of {n_shared} shared "
                f"(SNR_H1 ≥ {thr:g} dB)")
        return mismatched


    def _on_flow_mismatch_toggle(self, checked: bool):
        self._show_flow_mismatch = bool(checked)
        self._compute_flow_mismatch_edges()
        self._refresh_edges()


    def _on_flow_mismatch_snr_change(self, _v=None):
        self._compute_flow_mismatch_edges()
        if getattr(self, '_show_flow_mismatch', False):
            self._refresh_edges()


    def _on_source_change(self, _checked=None):
        # Four-way exclusive group; only act when EXACTLY one is checked
        radios = [self._src_radio_measured,
                  getattr(self, '_src_radio_rec', None),
                  self._src_radio_sim,
                  getattr(self, '_src_radio_sim_carved', None)]
        checked = [r for r in radios if r is not None and r.isChecked()]
        if len(checked) != 1:
            return  # transient state during exclusive-group switch
        if (getattr(self, '_src_radio_sim_carved', None) is not None
                and self._src_radio_sim_carved.isChecked()):
            new_src = 'sim_carved'
        elif self._src_radio_sim.isChecked():
            new_src = 'sim'
        elif (getattr(self, '_src_radio_rec', None) is not None
              and self._src_radio_rec.isChecked()):
            new_src = 'reconciled'
        else:
            new_src = 'measured'
        if new_src == self.current_source:
            return
        self.current_source = new_src
        # Re-sync enablement; the selection may need to fall back.
        self._sync_field_combos()
        # Refresh carve boundary markers if they're on (colours follow
        # the source).
        try:
            self._update_carve_boundary_markers()
        except Exception as _e:
            _note_suppressed("carve boundary markers on source change", _e)
        # Tile-harmonic cache must be rebuilt against the new source's Q_t
        if hasattr(self, '_tile_harmonic_cache'):
            self._tile_harmonic_cache.clear()
            if (getattr(self, 'current_tile_filter', None) is not None
                    and self.current_tile_filter != 'All'):
                self._populate_tile_harmonic_cache(self.current_tile_filter)
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()


    _IN_MEMORY_PREFIXES = ('_rec_',)


    def _dump_graph(self, f, protocol=None):
        """Pickle self.G without the viewer's in-memory reconciliation attrs (`_h_*_rec`, `_rec_*`)."""
        import pickle as _pk
        stash = []
        for u, v, d in self.G.edges(data=True):
            ks = [k for k in d if k.startswith(self._IN_MEMORY_PREFIXES) or (k.startswith('_h_') and k.endswith('_rec'))]
            if ks:
                stash.append((d, {k: d.pop(k) for k in ks}))
        try:
            if protocol is None:
                _pk.dump(self.G, f)
            else:
                _pk.dump(self.G, f, protocol=protocol)
        finally:
            for d, kv in stash:
                d.update(kv)


    def _require_workbench(self, what: str) -> bool:
        """Guard for anything that writes or recomputes; the read-only viewer refuses."""
        if self.workbench:
            return True
        print(f"  (read-only viewer: {what} needs `plexusflow workbench <stage>`)")
        return False


    def _apply_capabilities(self):
        """Hide the controls that write or recompute when not in workbench mode."""
        if self.workbench:
            return
        for name in ('_click_to_tag_chk', '_radial_btn_row', '_region_row',
                     '_run_of_btn', '_of_merge_chk', '_of_refine_chk', '_clear_of_region_btn'):
            w = getattr(self, name, None)
            if w is not None:
                w.setVisible(False)


    def _reconciliation_available(self) -> bool:
        """True if reconciliation outputs were found and attached to the graph."""
        return bool(getattr(self, '_rec', None))


    def _results_dir(self):
        """results/ next to the graph's stage (or $PLEXUSFLOW_RESULTS when it points at this stage's results)."""
        import os
        stage = self.graph_path.parent.parent
        env = os.environ.get('PLEXUSFLOW_RESULTS')
        cands = ([Path(env).expanduser()] if env else []) + [stage / 'results']
        for c in cands:
            if (c / 'dc_reconciled_v2.pkl').exists():
                return c
        return None


    def _load_reconciliation(self) -> dict:
        """Attach the reconciliation outputs to the in-memory graph (never written back):
        `_h_amp_DC_rec` (signed Q\u2032 along the stored edge orientation), `_rec_Qp_sd`, `_rec_unstable`,
        `_h_{amp,phase}_H{k}_rec` (tile-aligned AC field, eligible edges), `_rec_tile`, `_rec_gauge_sd`,
        `_rec_phase_ok` (winner tile gauge SD ≤ 30°).  Returns a small summary dict ({} if absent)."""
        res = self._results_dir()
        if res is None:
            return {}
        import pickle as _pk
        try:
            D = _pk.loads((res / 'dc_reconciled_v2.pkl').read_bytes())
            A = _pk.loads((res / 'ac_field_v2.pkl').read_bytes()) if (res / 'ac_field_v2.pkl').exists() else None
        except Exception as exc:
            print(f"  reconciliation: could not read {res}: {exc}")
            return {}
        edges = [tuple(e) for e in D['edges']]
        missing = sum(1 for (u, v) in edges if not self.G.has_edge(u, v))
        if missing:
            print(f"  reconciliation: {missing} of {len(edges)} edges not in this graph — results belong to "
                  f"another graph version; 'Reconciled' source disabled")
            return {}
        Qp = np.asarray(D['Qp'], float)
        sd = np.asarray(D.get('Qp_sd', np.full(len(edges), np.nan)), float)
        uns = np.asarray(D.get('unstable', np.zeros(len(edges), bool)), bool)
        for i, (u, v) in enumerate(edges):
            d = self.G.edges[u, v]
            d['_h_amp_DC_rec'] = float(Qp[i]); d['_rec_Qp_sd'] = float(sd[i]); d['_rec_unstable'] = bool(uns[i])
        n_ac = n_ph = 0
        f_ref = float('nan')
        if A is not None and [tuple(e) for e in A['edges']] == edges:
            from ..ac.field import phase_determined
            ph_ok = phase_determined(A, 30.0)
            gsd = A.get('gauge_sd_deg') or {}
            f_ref = float(A.get('f_ref', float('nan')))
            for i, (u, v) in enumerate(edges):
                if not A['elig'][i]:
                    continue
                d = self.G.edges[u, v]; t = int(A['tile'][i])
                for k in range(A['H'].shape[0]):
                    h = complex(A['H'][k, i])
                    d[f'_h_amp_H{k + 1}_rec'] = abs(h); d[f'_h_phase_H{k + 1}_rec'] = float(np.angle(h))
                d['_rec_tile'] = t; d['_rec_gauge_sd'] = float(gsd.get(t, np.nan)); d['_rec_phase_ok'] = bool(ph_ok[i])
                n_ac += 1; n_ph += int(ph_ok[i])
        print(f"  reconciliation ({res}): DC on {len(edges)} carved edges ({int(uns.sum())} flagged unstable); "
              f"AC field on {n_ac} edges, phase-determined {n_ph}")
        return dict(results_dir=res, n_dc=len(edges), n_ac=n_ac, n_phase=n_ph, f_ref=f_ref,
                    metrics=D.get('metrics', {}))


    def _on_nodes_toggle(self, checked: bool):
        self.show_nodes = bool(checked)
        self._refresh_nodes()


    def _on_tile_boundaries_toggle(self, checked: bool):
        self.show_tile_boundaries = bool(checked)
        self._refresh_tile_boundaries()


    def _on_tile_labels_toggle(self, checked: bool):
        self.show_tile_labels = bool(checked)
        self._refresh_tile_labels()


    def _on_edge_labels_toggle(self, checked: bool):
        self.show_edge_labels = bool(checked)
        self._refresh_edge_labels()


    def _on_kcl_toggle(self, checked: bool):
        self.show_kcl = bool(checked)
        self._refresh_kcl_overlay()


    def _refresh_kcl_overlay(self):
        """Show per-node KCL residual (signed Σ Q at the selected tile)
        as a Points layer.  Only renders when show_kcl is True AND a
        single tile is selected (current_tile_filter is not None)."""
        # Tear down any existing overlay first.
        if self._kcl_layer is not None:
            try:
                self.viewer.layers.remove(self._kcl_layer)
            except (KeyError, ValueError):
                pass
            self._kcl_layer = None
        if not self.show_kcl:
            return
        tile_id = self.current_tile_filter
        if tile_id is None:
            self._info_box.setPlainText(
                "Kirchhoff residuals need a specific tile selected "
                "(Tile filter in the View tab).")
            return

        # Compute per-node signed Q sum from THIS tile's measurements.
        residual = {n: 0.0 for n in self.G.nodes()}
        n_contrib = {n: 0 for n in self.G.nodes()}
        for u, v in self.G.edges():
            ed = self.G.edges[u, v]
            m = next((mm for mm in (ed.get('measurements_piv') or [])
                       if isinstance(mm, dict)
                       and MeasurementView(mm).tile_id == int(tile_id)), None)
            if m is None:
                continue
            mv_m = MeasurementView(m)
            mq = m.get('Q_DC')
            if mq is None or not np.isfinite(mq):
                # v4 migration: was m.get('mean_Q')
                mq = mv_m.harm.mean_Q
            if mq is None or not np.isfinite(mq):
                continue
            # v4 migration: was m.get('flow_from') / m.get('flow_to')
            ff = mv_m.flow_from; ft = mv_m.flow_to
            if ff is None or ft is None:
                continue
            # Treat measured |Q| as flowing ff → ft.  Signed residual at
            # node n = (in − out) so positive = net inflow, negative =
            # net outflow.  By KCL on an interior node, residual ≈ 0.
            q = float(abs(mq))
            # Flow leaves ff, arrives at ft.  Subtract at ff, add at ft.
            residual[ff] -= q
            residual[ft] += q
            n_contrib[ff] += 1
            n_contrib[ft] += 1

        # Build node positions + colour.
        nodes_with_data = [n for n, c in n_contrib.items() if c > 0]
        if not nodes_with_data:
            return
        pts = np.array([[self.mosaic_height - self.G.nodes[n]['y'],
                          self.G.nodes[n]['x']]
                         for n in nodes_with_data])
        vals = np.array([residual[n] for n in nodes_with_data])
        max_abs = float(np.percentile(np.abs(vals), 99)) or 1.0
        # Diverging colormap: red = positive (net inflow), blue = negative.
        from matplotlib import cm as _mpl_cm
        norm = mcolors.Normalize(vmin=-max_abs, vmax=+max_abs)
        colors = _mpl_cm.RdBu_r(norm(vals))
        # Size by |residual| relative to max, so the worst residuals pop.
        sizes = 6.0 + 20.0 * np.abs(vals) / max(max_abs, 1e-12)
        self._kcl_layer = self.viewer.add_points(
            pts, name=f"KCL residual (tile {tile_id})  ·  "
                       f"max |Σ Q| = {max_abs:.3f} nL/s",
            face_color=colors, size=sizes,
            edge_color='black', edge_width=0.4,
            opacity=0.85,
        )
        # Diagnostic summary in the info box.
        interior_vals = []
        boundary_vals = []
        for n in nodes_with_data:
            if self.G.nodes[n].get('boundary_type') in ('source', 'sink'):
                boundary_vals.append(residual[n])
            else:
                interior_vals.append(residual[n])
        msg_parts = [f"Tile {tile_id} KCL residual:"]
        if interior_vals:
            interior_vals = np.array(interior_vals)
            msg_parts.append(
                f"  interior (n={len(interior_vals)}): "
                f"RMS = {float(np.sqrt(np.mean(interior_vals**2))):.4f} nL/s, "
                f"max |Σ| = {float(np.abs(interior_vals).max()):.4f}")
        if boundary_vals:
            boundary_vals = np.array(boundary_vals)
            msg_parts.append(
                f"  boundary (n={len(boundary_vals)}): "
                f"max |Σ| = {float(np.abs(boundary_vals).max()):.4f}  "
                f"(expected non-zero — net carve in/outflow)")
        self._info_box.setPlainText("\n".join(msg_parts))

        # Popup the distribution histogram.
        self._render_kcl_distribution(
            tile_id,
            np.asarray(interior_vals, dtype=float),
            np.asarray(boundary_vals, dtype=float),
        )


    def _render_kcl_distribution(self, tile_id, interior_vals, boundary_vals):
        """Save + display a histogram of KCL residuals for the current
        tile.  Splits interior (should be ≈ 0) from boundary (carve net
        flow) so they can be compared on the same axes."""
        import matplotlib.pyplot as plt
        import tempfile
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        # Pool both sets for x-range determination.
        pool = []
        if interior_vals.size: pool.append(interior_vals)
        if boundary_vals.size: pool.append(boundary_vals)
        if not pool:
            plt.close(fig); return
        all_vals = np.concatenate(pool)
        lim = float(max(np.percentile(np.abs(all_vals), 99) * 1.15, 1e-6))
        bins = np.linspace(-lim, lim, 50)
        if interior_vals.size:
            rms_i = float(np.sqrt(np.mean(interior_vals ** 2)))
            ax.hist(interior_vals, bins=bins, density=False,
                    histtype='stepfilled', color='#4477aa', alpha=0.7,
                    edgecolor='#1f3a5a', linewidth=1.2,
                    label=f"interior  (n={len(interior_vals)},  RMS={rms_i:.4f} nL/s)")
        if boundary_vals.size:
            ax.hist(boundary_vals, bins=bins, density=False,
                    histtype='stepfilled', color='#cc6677', alpha=0.7,
                    edgecolor='#7a2a32', linewidth=1.2,
                    label=f"boundary  (n={len(boundary_vals)})")
        ax.axvline(0.0, color='#333', lw=0.6, ls=':')
        ax.set_xlabel("KCL residual  (Σ Q arriving − Σ Q leaving)  [nL/s]")
        ax.set_ylabel("node count")
        ax.set_title(f"KCL residual distribution  ·  tile {tile_id}")
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.25)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            png = f.name
        fig.savefig(png, dpi=130, bbox_inches='tight')
        plt.close(fig)
        # Display in a small Qt dialog.
        try:
            from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea
            from qtpy.QtGui import QPixmap
            from qtpy.QtCore import Qt
            dlg = QDialog(self.viewer.window._qt_window)
            dlg.setWindowTitle(f"KCL residual distribution — tile {tile_id}")
            lay = QVBoxLayout(dlg)
            scroll = QScrollArea(); scroll.setWidgetResizable(True)
            lbl = QLabel()
            pm = QPixmap(png); lbl.setPixmap(pm); lbl.setAlignment(Qt.AlignCenter)
            scroll.setWidget(lbl); lay.addWidget(scroll)
            dlg.resize(min(int(pm.width()) + 30, 1200),
                        min(int(pm.height()) + 80, 800))
            self._kcl_dist_dialog = dlg   # retain so GC doesn't kill it
            dlg.show()
        except Exception as e:
            self._info_box.append(f"\n(could not show histogram dialog: {e})\n"
                                    f"figure saved to: {png}")


    def _refresh_edge_labels(self):
        """Render the current Property value as text at each edge's
        midpoint.  Only active when self.show_edge_labels is True AND
        Tile filter is set to a specific tile — in All-tiles mode the
        labels clutter the mosaic.
        """
        # Always start by removing any existing label layer; we then
        # decide whether to add a fresh one based on current state.
        if self._edge_labels_layer is not None:
            try:
                self.viewer.layers.remove(self._edge_labels_layer)
            except (KeyError, ValueError):
                pass
            self._edge_labels_layer = None
        if not self.show_edge_labels:
            return
        if self.current_tile_filter is None:
            return
        tile_id = int(self.current_tile_filter)
        prop = self.current_property
        # Format helpers for the most common property types
        def _fmt(val: float) -> str:
            if prop == 'phase':
                return f"{np.degrees(val):+.0f}°"
            if prop in ('snr', 'snr_ac', 'snr_total'):
                return f"{val:+.1f} dB"
            if prop == 'PI':
                return f"{val:.2f}"
            if prop == 'frequency':
                return f"{val:.2f} Hz"
            if prop == 'dissipation':
                return f"{val:.1e} W"
            if prop in ('radius', 'length'):
                return f"{val:.0f}"
            # default: 3-sig-fig
            return f"{val:.3g}"
        points = []
        texts = []
        H_mosaic = self.mosaic_height
        for u, v in self.edge_list:
            d = self.G.edges[u, v]
            # Only edges with a measurement on the active tile.
            # v4 migration: was m.get('tile_id')
            if not any(MeasurementView(m).tile_id == tile_id
                       for m in (d.get('measurements_piv') or [])):
                continue
            val = self._resolve_field_value(u, v)
            if val is None or not np.isfinite(val):
                continue
            x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
            x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            xm = 0.5 * (float(x1) + float(x2))
            ym = 0.5 * (float(y1) + float(y2))
            points.append([H_mosaic - ym, xm])
            texts.append(_fmt(float(val)))
        if not points:
            return
        try:
            self._edge_labels_layer = self.viewer.add_points(
                np.asarray(points, dtype=float),
                size=0,
                text={'string': texts, 'size': 8, 'color': 'white',
                       'anchor': 'center'},
                name='Edge labels',
                face_color='transparent',
                border_color='transparent',
            )
        except Exception as e:
            print(f"  Could not add edge-label layer: {e}")


    def _load_video_for_selected(self):
        tid = getattr(self, '_selected_tile_id', None)
        if tid is None:
            return
        self._load_video_overlay(int(tid))
        self._unload_video_btn.setEnabled(self._video_layer is not None)


    def _load_video_from_dropdown(self):
        if not hasattr(self, '_tile_combo'):
            return
        # currentData() returns the int tile ID stored as userData;
        # falls back to None for the "(pick a tile)" placeholder and
        # for greyed-out entries (which can't be selected anyway).
        tid = self._tile_combo.currentData()
        if tid is None:
            return
        self._load_video_overlay(int(tid))
        self._unload_video_btn.setEnabled(self._video_layer is not None)


    # ── public entry ───────────────────────────────────────────────────
    def run(self):
        self._setup_viewer()
        self._setup_panel()
        self._refresh_edges()
        self._refresh_nodes()
        # Initial colour-bar draw — needs `_last_vmin`/`_last_vmax`
        # populated by the first `_refresh_edges()` above, so do it here
        # rather than at the end of `_setup_panel`.
        self._refresh_cbar()
        if self.rotate_cw or self.rotate_ccw:
            self._apply_view_rotation(ccw=self.rotate_ccw)
        import napari
        napari.run()


    def _apply_view_rotation(self, ccw: bool = False):
        """Rotate every layer by 90° on top of its base affine.

        CW   (default):  (row, col) → (col,         H − row)
        CCW (ccw=True):  (row, col) → (W − col,     row     )

        Useful for datasets stored on their sides (e.g. 15-somite,
        which needs CCW rotation to put the embryo body axis vertical
        in the same orientation as the 21- and 27-somite mosaics).

        Caches the rotation affine and subscribes to the layers'
        `events.inserted` signal so that layers added LATER (tile
        numbers, video overlay, click markers, …) also get rotated.
        """
        H = self.mosaic.shape[0] if self.mosaic is not None else int(
            self.G.graph.get('mosaic_height', 0))
        W = self.mosaic.shape[1] if self.mosaic is not None else int(
            self.G.graph.get('mosaic_width', 0))
        if H <= 0 or W <= 0:
            return
        if ccw:
            rot_affine = np.array([[0, -1, W],
                                     [1,  0, 0],
                                     [0,  0, 1]], dtype=float)
        else:
            rot_affine = np.array([[0,  1, 0],
                                     [-1, 0, H],
                                     [0,  0, 1]], dtype=float)
        self._rotation_affine = rot_affine

        # Only skip the Mosaic image — its tissue is already in the
        # correct/upright orientation.  Everything else (graph shapes,
        # points, markers, per-tile Video overlays) was captured in a
        # 90°-CCW frame relative to the mosaic, so display-rotating them
        # aligns them with the upright tissue.
        _skip_prefixes = ('Mosaic',)

        def _should_skip(layer) -> bool:
            name = str(getattr(layer, 'name', '') or '')
            return any(name.startswith(p) for p in _skip_prefixes)

        def _rotate_layer(layer):
            if _should_skip(layer):
                return
            base = np.array(layer.affine.affine_matrix)
            if base.shape[0] == 3:
                layer.affine = rot_affine @ base
            elif base.shape[0] == 4:
                rot4 = np.eye(4)
                rot4[1:3, 1:3] = rot_affine[:2, :2]
                rot4[1:3, 3] = rot_affine[:2, 2]
                layer.affine = rot4 @ base

        for layer in self.viewer.layers:
            _rotate_layer(layer)

        # Catch layers added later (tile-number overlay, video, markers).
        def _on_layer_inserted(event):
            try:
                _rotate_layer(event.value)
            except Exception as exc:
                print(f"  WARN: failed to rotate new layer "
                      f"{getattr(event.value, 'name', '?')!r}: {exc}")
        self.viewer.layers.events.inserted.connect(_on_layer_inserted)


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def main():
    # Strip empty-string CLI args — VS Code launch.json inputs pass
    # through an empty "" placeholder for optional flags (e.g. a
    # conditional --rotate-cw); argparse would reject "" otherwise.
    sys.argv = [a for a in sys.argv if a != '']

    parser = argparse.ArgumentParser(
        description="Read-only mosaic viewer (intern + publication use).")
    parser.add_argument('graph', nargs='?', default=None,
                        help="Path to mosaic_graph_analyzed.gpickle. "
                             "Optional if --config is given.")
    parser.add_argument('--config', default=None,
                        help="Path to a batch_analyze_config.json (the "
                             "same one the analysis pipeline uses).  "
                             "Convenience: pulls graph, tiff, tile "
                             "positions, video dir, and video pattern "
                             "from one file.  Individual --tiff / "
                             "--tile-positions / --video-dir / "
                             "--video-pattern args override.")
    parser.add_argument('--tiff', default=None,
                        help="Optional stitched mosaic TIFF for background.")
    parser.add_argument('--tile-positions', default=None,
                        help="Path to tile_positions_manual.json.  "
                             "Required for video overlay.")
    parser.add_argument('--video-dir', default=None, action='append',
                        help="Directory containing per-tile videos. Required "
                             "for video overlay.  May be given MULTIPLE times "
                             "(or comma-separated) to list candidate locations "
                             "for different machines; the first that exists on "
                             "this machine is used (e.g. a desktop path and a "
                             "laptop path in one command).")
    parser.add_argument('--video-pattern', default=None,
                        help="Filename pattern for tile videos, e.g. "
                             "'10x 250fps loc{vid}_C001H001S0001'. "
                             "Required for video overlay.")
    parser.add_argument('--of-video-dir', default=None, action='append',
                        help="Optional directory for fresh OF/PIV compute. "
                             "Use this to display downsampled videos while "
                             "computing on raw frame folders. Falls back to "
                             "--video-dir when omitted.")
    parser.add_argument('--of-video-pattern', default=None,
                        help="Optional filename/folder pattern for fresh "
                             "OF/PIV compute videos. Falls back to "
                             "--video-pattern when omitted.")
    parser.add_argument('--initial-field', default='mean_Q',
                        help="Field to colour edges by on startup. "
                             "Default: mean_Q.")
    parser.add_argument('--no-cache-harmonic-class', action='store_true',
                        help="Don't write the precomputed harmonic_class "
                             "back to the gpickle.  Default is to cache "
                             "(adds an edge attr; touches no other data).")
    parser.add_argument('--force-recompute-harmonic-class',
                        action='store_true',
                        help="Wipe any existing harmonic_class values and "
                             "recompute from scratch.  Useful if PIV was "
                             "re-run and the cached values are stale.")
    parser.add_argument('--workbench', action='store_true',
                        help="Editing tools (fresh OF, tagging, boundary edits, saves); default is read-only view.")
    parser.add_argument('--rotate-cw', action='store_true',
                        help="Rotate the graph 90° clockwise on startup.")
    parser.add_argument('--rotate-ccw', action='store_true',
                        help="Rotate the graph 90° counter-clockwise on "
                             "startup.  Use for 15-somite (stored on its "
                             "side).")
    parser.add_argument('--auto-rotate', action='store_true',
                        help="Auto-select rotation from graph path: "
                             "'Somites21' → --rotate-cw, 'Somites15' → "
                             "--rotate-ccw, others → no rotation.")
    args = parser.parse_args()

    # Auto-rotate: derive from graph path if no explicit flag was given.
    if args.auto_rotate and not (args.rotate_cw or args.rotate_ccw):
        graph_str = str(args.graph or '')
        if 'Somites21' in graph_str:
            args.rotate_cw = True
        elif 'Somites15' in graph_str:
            args.rotate_ccw = True

    # Resolve paths via config if provided; individual args win.
    # Paths in the config can be RELATIVE to the config file's directory
    # — that makes the data folder portable (intern only changes the
    # config path, not its contents).
    cfg: dict = {}
    cfg_dir: Optional[Path] = None
    if args.config:
        cfg_path = Path(args.config).resolve()
        cfg_dir = cfg_path.parent
        with open(cfg_path) as f:
            cfg = json.load(f)

    def _resolve(cli_val, cfg_key):
        v = cli_val if cli_val is not None else cfg.get(cfg_key)
        if v is None:
            return None
        p = Path(v)
        if p.is_absolute() or cfg_dir is None:
            return p
        return (cfg_dir / p).resolve()

    def _resolve_multi(cli_list, cfg_key):
        """Resolve a video-dir spec to a LIST of candidate paths (CLI args win
        over config; config value may itself be a string or a list).  Each
        relative path is resolved against the config dir."""
        vals = cli_list if cli_list else cfg.get(cfg_key)
        if vals is None:
            return None
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        out = []
        for v in vals:
            parts = str(v).split(',') if isinstance(v, str) else [v]
            for part in parts:
                s = str(part).strip()
                if not s:
                    continue
                p = Path(s)
                out.append(p if (p.is_absolute() or cfg_dir is None)
                           else (cfg_dir / p).resolve())
        return out or None

    # Prefer a viewer-specific graph key so the config's `mosaic_graph`
    # (the BATCH pipeline's input) isn't loaded by the viewer.  Falls back
    # to `mosaic_graph` for old configs.  CLI positional always wins.
    graph = _resolve(args.graph, 'viewer_graph')
    if not graph:
        graph = _resolve(args.graph, 'mosaic_graph')
    if not graph:
        parser.error("graph path required (positional arg or "
                     "config['viewer_graph'] / config['mosaic_graph']).")
    tiff = _resolve(args.tiff, 'mosaic_tiff')
    tile_positions = _resolve(args.tile_positions, 'tile_positions')
    video_dir = _resolve_multi(args.video_dir, 'video_dir')
    of_video_dir = _resolve_multi(args.of_video_dir, 'of_video_dir')
    # video_pattern is a filename pattern (not a path) — pass through.
    video_pattern = args.video_pattern or cfg.get('video_pattern')
    of_video_pattern = (args.of_video_pattern
                        or cfg.get('of_video_pattern')
                        or video_pattern)

    viewer = ReadOnlyMosaicViewer(
        Path(graph),
        tiff_path=Path(tiff) if tiff else None,
        tile_positions=Path(tile_positions) if tile_positions else None,
        video_dir=video_dir,            # list of candidate dirs (or None)
        video_pattern=video_pattern,
        of_video_dir=of_video_dir,      # list of candidate dirs (or None)
        of_video_pattern=of_video_pattern,
        initial_field=args.initial_field,
        cache_harmonic_class=not args.no_cache_harmonic_class,
        mode='workbench' if args.workbench else 'view',
        force_recompute_harmonic_class=args.force_recompute_harmonic_class,
        rotate_cw=args.rotate_cw,
        rotate_ccw=args.rotate_ccw,
    )
    viewer.run()


if __name__ == '__main__':
    main()
