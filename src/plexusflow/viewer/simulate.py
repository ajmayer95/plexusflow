"""Simulate tab: carved-network and transmission-line simulations, pump, boundary-condition waveforms.

Mixin of plexusflow.viewer.app.ReadOnlyMosaicViewer (methods moved verbatim from app.py)."""
from __future__ import annotations

from ._common import (  # noqa: F401
    Dict,
    List,
    SIM_FIELD_MAP,
    SIM_INTERNAL_KEYS,
    Tuple,
    _actual_boundary_Q_dc,
    _print_dc_bc_diagnostic,
    np,
)


class SimulateMixin:

    # ─────────────────────────────────────────────────────────────────
    # =================================================================
    # Normalize tab: per-tile complex gauges + amp gauges, computed
    # in-memory via `pertile.mosaic.tile_normalize`.  Writes normalized
    # measurements side-by-side with raw and unlocks the 'Normalized'
    # data source in the View tab.
    # =================================================================
    def _build_carved_sim_tab(self, layout):
        """Panel controls for the porous-carved transmission-line sim.

        Runs the recipe:
          1. Recompute simple-normalization v2 fields (with optional
             tile exclusion).
          2. Fit per-outlet α_k on the DC field via constrained NNLS.
          3. Carve the graph (drop tagged edges + A anchors, promote
             outlets to sources), extract DA reference + per-outlet
             local phase.
          4. Run the LOCKED two-pass TL sim (P=0 at V pass 1, Q-rebuild
             at V pass 2) with the outlets as sources.
          5. Write `_h_*_sim_carved` fields the View tab reads.
        """
        from qtpy.QtWidgets import (
            QLabel, QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout,
            QLineEdit, QDoubleSpinBox, QPlainTextEdit, QCheckBox)
        # ── Detect status (updated on each refresh) ─────────────────
        _stat_group = QGroupBox("Detected carve")
        _stat_lay = QVBoxLayout(_stat_group)
        self._carved_stat_lbl = QLabel(self._carved_sim_status_text())
        self._carved_stat_lbl.setWordWrap(True)
        self._carved_stat_lbl.setStyleSheet(
            "color: #aaaaaa; font-size: 11px;")
        _stat_lay.addWidget(self._carved_stat_lbl)
        _refresh_btn = QPushButton("Refresh status")
        _refresh_btn.clicked.connect(self._carved_sim_refresh_status)
        _stat_lay.addWidget(_refresh_btn)
        layout.addWidget(_stat_group)

        # ── Inputs ──────────────────────────────────────────────────
        _in_group = QGroupBox("Sim inputs")
        _in_lay = QVBoxLayout(_in_group)
        _row = QHBoxLayout()
        _row.addWidget(QLabel("Exclude tiles:"))
        self._carved_excl_edit = QLineEdit("")
        self._carved_excl_edit.setPlaceholderText(
            "comma-separated (e.g. 1,2,3,4,5,6 for the low-f0 ramp-up tiles)")
        self._carved_excl_edit.setToolTip(
            "Which per-tile measurements to skip when rebuilding the v2 "
            "fields, computing per-outlet phase shifts, and fitting α.  "
            "Tiles 1-6 are the developmental HR ramp-up recordings.")
        _row.addWidget(self._carved_excl_edit)
        _in_lay.addLayout(_row)

        _row = QHBoxLayout()
        _row.addWidget(QLabel("D_areal [1/Pa]:"))
        self._carved_D_spin = QDoubleSpinBox()
        self._carved_D_spin.setDecimals(6)
        self._carved_D_spin.setRange(1e-7, 1e-2)
        self._carved_D_spin.setSingleStep(1e-5)
        self._carved_D_spin.setValue(3.5e-4)
        self._carved_D_spin.setToolTip(
            "Areal distensibility used by the transmission-line solver.")
        _row.addWidget(self._carved_D_spin)
        _in_lay.addLayout(_row)

        _row = QHBoxLayout()
        _row.addWidget(QLabel("η (loss tan):"))
        self._carved_eta_spin = QDoubleSpinBox()
        self._carved_eta_spin.setDecimals(2)
        self._carved_eta_spin.setRange(0.0, 2.0)
        self._carved_eta_spin.setSingleStep(0.05)
        self._carved_eta_spin.setValue(0.0)
        self._carved_eta_spin.setToolTip(
            "Wall loss tangent η in D_eff(ω) = D·(1 − iη).\n"
            "0 = elastic (default). ~0.4 matches the current fit sweet spot.")
        _row.addWidget(self._carved_eta_spin)
        _row.addStretch(1)
        _in_lay.addLayout(_row)

        _row = QHBoxLayout()
        _row.addWidget(QLabel("V handling:"))
        _lbl = QLabel("LOCKED two-pass (P=0 pass 1, Q-rebuild pass 2)")
        _lbl.setStyleSheet("color: #666; font-style: italic;")
        _row.addWidget(_lbl); _row.addStretch(1)
        _in_lay.addLayout(_row)

        _row = QHBoxLayout()
        _row.addWidget(QLabel("α source:"))
        _lbl = QLabel("Constrained NNLS on measured DC field (Σα_zone = equal)")
        _lbl.setStyleSheet("color: #666; font-style: italic;")
        _row.addWidget(_lbl); _row.addStretch(1)
        _in_lay.addLayout(_row)
        layout.addWidget(_in_group)

        # ── Run ─────────────────────────────────────────────────────
        _run_group = QGroupBox("Run")
        _run_lay = QVBoxLayout(_run_group)
        self._carved_run_btn = QPushButton("Run Carved Sim")
        self._carved_run_btn.setToolTip(
            "Rebuilds v2 fields, fits α_k, runs the LOCKED two-pass TL sim, "
            "and writes _h_*_sim_carved fields.  Takes ~30-60 s.")
        self._carved_run_btn.clicked.connect(self._carved_sim_run)
        _run_lay.addWidget(self._carved_run_btn)
        self._carved_status = QPlainTextEdit()
        self._carved_status.setReadOnly(True)
        self._carved_status.setMaximumHeight(220)
        self._carved_status.setStyleSheet(
            "font-family: monospace; font-size: 11px;")
        _run_lay.addWidget(self._carved_status)
        layout.addWidget(_run_group)

        # ── View overlay ────────────────────────────────────────────
        _view_group = QGroupBox("Display")
        _view_lay = QVBoxLayout(_view_group)
        self._carved_use_chk = QCheckBox(
            "Show the carved sim on the View tab (_h_*_sim_carved fields)")
        self._carved_use_chk.setToolTip(
            "When on, the View tab reads _h_*_sim_carved fields.  When off, "
            "the normal `_h_*` / `_h_*_norm` fields are used.")
        self._carved_use_chk.toggled.connect(
            self._carved_sim_view_toggled)
        _view_lay.addWidget(self._carved_use_chk)
        self._carve_markers_chk = QCheckBox(
            "Show carve boundary markers  (★ = flow OUT of porous, "
            "□ = flow IN to porous)")
        self._carve_markers_chk.setToolTip(
            "Overlays a marker at each outlet node, coloured by the "
            "current field (matches edge colouring).  Star = net flow "
            "OUT of the porous zone into the trusted plexus; Square = "
            "net flow INTO the porous zone from the plexus.")
        self._carve_markers_chk.toggled.connect(
            self._on_carve_markers_toggled)
        _view_lay.addWidget(self._carve_markers_chk)
        layout.addWidget(_view_group)


    def _carved_sim_status_text(self):
        """One-line status describing the carve on the current graph."""
        try:
            tagged = sum(1 for _, _, d in self.G.edges(data=True)
                          if d.get('radial_uncertain'))
            if tagged == 0:
                return ("No `radial_uncertain` tags on the graph — nothing "
                        "to carve.  Tag some vessels via the View tab first.")
            from plexusmech.viewer.carved_sim_backend import find_porous_and_outlets
            U, outlets, outlet_comp, A_a, V_a, comps = find_porous_and_outlets(
                self.G)
            comp_sizes = [len(c) for c in comps]
            return (f"tagged edges: {tagged}   porous nodes: {len(U)}   "
                    f"outlets: {len(outlets)}   "
                    f"zones: {len(comps)} sizes {comp_sizes}\n"
                    f"A anchors: {sorted(A_a)}   V anchors: {sorted(V_a)}")
        except Exception as e:
            return f"(status error: {e})"


    def _carved_sim_refresh_status(self):
        self._carved_stat_lbl.setText(self._carved_sim_status_text())


    def _carved_sim_view_toggled(self, on: bool):
        # Cache the flag; `_refresh_edges` reads it to decide which field
        # to use for edge colouring.
        self._carved_sim_show = bool(on)
        if hasattr(self, '_refresh_edges'):
            self._refresh_edges()


    def _on_carve_markers_toggled(self, on: bool):
        self._show_carve_markers = bool(on)
        self._update_carve_boundary_markers()


    def _classify_outlet_direction(self, o, U, A_a, V_a):
        """Return ('in', 1.0) if `o` is the porous-side node that a
        boundary vessel (A-anchor stub) feeds INTO the porous zone via
        — the physical inflow port for the region.  Otherwise return
        ('out', 1.0) because by construction of our sim, every other
        outlet is a source (α_k ≥ 0) delivering flow OUT of the porous
        zone into the trusted plexus.

        Detection: if `o` is a direct neighbor of any A anchor via an
        UNtagged edge (the A stub), it's the inflow port.  All other
        outlets are outflow."""
        for a in A_a:
            if not self.G.has_edge(o, a): continue
            d = self.G.edges[o, a]
            if d.get('radial_uncertain'): continue
            # o is on an A stub → inflow-from-A boundary
            return 'in', 1.0
        # V-anchor-adjacent porous nodes (unlikely in our current
        # geometry but handled for generality) — those would also be
        # inflow ports for the porous zone from the V side.
        for vn in V_a:
            if not self.G.has_edge(o, vn): continue
            d = self.G.edges[o, vn]
            if d.get('radial_uncertain'): continue
            return 'in', 1.0
        return 'out', 1.0


    def _update_carve_boundary_markers(self):
        """Add/update the 'CarveMarkers' napari points layer.  Each
        outlet gets a ★ (net flow OUT of porous), and every porous-side
        node of an A- or V-anchor boundary vessel gets a □ (inflow port
        from the anatomical anchor).  Colours match the current edge
        colormap.  No-op if the checkbox is off or nothing to show."""
        if not getattr(self, '_show_carve_markers', False):
            if 'CarveMarkers' in self.viewer.layers:
                self.viewer.layers['CarveMarkers'].visible = False
            return
        try:
            from plexusmech.viewer.carved_sim_backend import find_porous_and_outlets
            U, outlets, _, A_a, V_a, _ = find_porous_and_outlets(self.G)
        except Exception:
            return
        # Also find porous-side nodes of A/V anchor stubs — the inflow
        # ports for the region.  These may or may not qualify as
        # "outlets" (they don't if their only untagged edge is TO the
        # anchor).  We mark them as ▢ regardless.
        inflow_nodes = set()
        for anc in list(A_a) + list(V_a):
            if not self.G.has_node(anc): continue
            for m in self.G.neighbors(anc):
                if m not in U: continue
                d = self.G.edges[m, anc]
                if d.get('radial_uncertain'): continue
                inflow_nodes.add(m)
        # Marker list = outlets ∪ inflow_nodes (inflow_nodes win on
        # symbol tie-break).
        marker_nodes = list(dict.fromkeys(list(outlets) + list(inflow_nodes)))
        if not marker_nodes:
            return
        pts = []; symbols = []; face = []
        # Get the current edge LC's cmap + norm so markers use the same
        # scaling.
        cmap_name = 'viridis'; norm = None
        if hasattr(self, '_edge_layer') and self._edge_layer is not None:
            try:
                cmap_name = self._edge_layer.face.colormap.name
            except Exception:
                pass
        # Build a colormap-lookup function from matplotlib.
        import matplotlib.cm as _mcm
        cmap_obj = _mcm.get_cmap(cmap_name)
        # Field values for markers — sampled from the "outward" edge
        # (outlet: untagged edge to trusted plexus; inflow node:
        # untagged edge to the anchor, i.e. the A/V stub itself).
        def _sample_val(n):
            # For an inflow port, pick the A/V-stub edge; for a normal
            # outlet, pick an untagged plexus-facing edge.
            if n in inflow_nodes:
                for anc in list(A_a) + list(V_a):
                    if self.G.has_edge(n, anc) and \
                            not self.G.edges[n, anc].get('radial_uncertain'):
                        if hasattr(self, '_resolve_field_value'):
                            v = self._resolve_field_value(n, anc)
                            if v is not None: return float(v)
                            v = self._resolve_field_value(anc, n)
                            if v is not None: return float(v)
            for m in self.G.neighbors(n):
                if m in U or m in A_a or m in V_a: continue
                if self.G.edges[n, m].get('radial_uncertain'): continue
                if hasattr(self, '_resolve_field_value'):
                    v = self._resolve_field_value(n, m)
                    if v is not None: return float(v)
                    v = self._resolve_field_value(m, n)
                    if v is not None: return float(v)
            return np.nan

        vals = np.asarray([_sample_val(n) for n in marker_nodes], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            v_lo, v_hi = float(np.nanpercentile(finite, 2)), \
                         float(np.nanpercentile(finite, 98))
        else:
            v_lo, v_hi = 0.0, 1.0
        vspan = max(v_hi - v_lo, 1e-9)

        H_mos = getattr(self, 'mosaic_height', None)
        for o, val in zip(marker_nodes, vals):
            if o in inflow_nodes:
                sym = 'square'
            else:
                sym = 'star'
            if np.isfinite(val):
                t = float(np.clip((val - v_lo) / vspan, 0.0, 1.0))
                rgba = list(cmap_obj(t))
            else:
                rgba = [0.5, 0.5, 0.5, 0.8]
            x = float(self.G.nodes[o]['x'])
            y = float(self.G.nodes[o]['y'])
            # napari points want (row, col) — same convention as the
            # existing Edges layer uses (row = H_mos - y).
            row = (float(H_mos) - y) if H_mos is not None else y
            pts.append([row, x]); symbols.append(sym); face.append(rgba)
        if not pts:
            return
        pts_np = np.asarray(pts, dtype=float)
        sym_np = np.asarray(symbols, dtype=object)
        face_np = np.asarray(face, dtype=float)
        if 'CarveMarkers' in self.viewer.layers:
            lyr = self.viewer.layers['CarveMarkers']
            lyr.data = pts_np
            try:
                lyr.symbol = sym_np
            except Exception:
                pass
            try:
                lyr.face_color = face_np
            except Exception:
                pass
            lyr.visible = True
        else:
            self.viewer.add_points(
                pts_np, name='CarveMarkers',
                symbol=sym_np,
                face_color=face_np,
                size=40,
                border_color='black', border_width=0.15,
            )

        # ── Boundary vessels (A/V stubs) as their own coloured shapes ─
        # These are the untagged edges from an A/V anchor to a porous
        # node.  Coloured the same way (current colormap) using the
        # `_resolve_field_value` on that edge.
        bv_lines = []; bv_cols = []
        for anc in list(A_a) + list(V_a):
            if not self.G.has_node(anc): continue
            for m in self.G.neighbors(anc):
                if m not in U: continue
                d = self.G.edges[m, anc]
                if d.get('radial_uncertain'): continue
                x1 = float(self.G.nodes[anc]['x']); y1 = float(self.G.nodes[anc]['y'])
                x2 = float(self.G.nodes[m]['x']);   y2 = float(self.G.nodes[m]['y'])
                row1 = (float(H_mos) - y1) if H_mos is not None else y1
                row2 = (float(H_mos) - y2) if H_mos is not None else y2
                bv_lines.append([[row1, x1], [row2, x2]])
                val = None
                if hasattr(self, '_resolve_field_value'):
                    val = self._resolve_field_value(anc, m)
                    if val is None:
                        val = self._resolve_field_value(m, anc)
                if val is not None and np.isfinite(val):
                    t = float(np.clip((float(val) - v_lo) / vspan, 0.0, 1.0))
                    bv_cols.append(list(cmap_obj(t)))
                else:
                    bv_cols.append([0.9, 0.6, 0.1, 0.9])   # fallback amber
        if bv_lines:
            bv_arr = np.asarray(bv_lines, dtype=float)
            bv_col_arr = np.asarray(bv_cols, dtype=float)
            if 'CarveBoundaryVessels' in self.viewer.layers:
                lyr = self.viewer.layers['CarveBoundaryVessels']
                lyr.data = list(bv_arr)
                try: lyr.edge_color = bv_col_arr
                except Exception:
                    try: lyr.border_color = bv_col_arr
                    except Exception: pass
                lyr.visible = True
            else:
                self.viewer.add_shapes(
                    list(bv_arr), shape_type='line',
                    name='CarveBoundaryVessels',
                    edge_color=bv_col_arr,
                    edge_width=8.0)
        else:
            if 'CarveBoundaryVessels' in self.viewer.layers:
                self.viewer.layers['CarveBoundaryVessels'].visible = False


    def _carved_sim_run(self):
        from qtpy.QtWidgets import QApplication
        import time
        self._carved_run_btn.setEnabled(False)
        self._carved_status.setPlainText("Starting ...")
        QApplication.processEvents()
        # Parse exclude-tiles text
        excl = set()
        txt = self._carved_excl_edit.text().strip()
        if txt:
            for t in txt.split(','):
                t = t.strip()
                if not t: continue
                try:
                    excl.add(int(t))
                except ValueError:
                    pass
        D = float(self._carved_D_spin.value())
        eta = float(getattr(self, '_carved_eta_spin', None).value()
                     if getattr(self, '_carved_eta_spin', None) is not None else 0.0)
        # Install η monkey-patch (constant-loss-tangent form D_eff = D·(1 − iη))
        from plexusmech.tl import transmission_line as _tl_c
        if not hasattr(type(self), '_pristine_vessel_admittance'):
            type(self)._pristine_vessel_admittance = _tl_c._vessel_admittance
        _orig_va_c = type(self)._pristine_vessel_admittance
        _eta_c = eta
        def _patched_va_c(R_m, L_m, omega, mu=_tl_c.MU_DEFAULT,
                           rho=_tl_c.RHO_BLOOD, D=None):
            D_eff = (D if D is not None else 1e-6) * (1.0 - 1j * _eta_c)
            return _orig_va_c(R_m, L_m, omega, mu, rho, D_eff)
        _tl_c._vessel_admittance = _patched_va_c
        try:
            from plexusmech.viewer.carved_sim_backend import run_carved_sim_full
            t0 = time.time()
            self._carved_status.appendPlainText(
                f"Excluding tiles: {sorted(excl) if excl else 'none'}\n"
                f"D_areal = {D:.2e}   η = {eta:.2f}\n"
                f"Running (see terminal for solver progress) ...")
            QApplication.processEvents()
            stats = run_carved_sim_full(
                self.G, exclude_tiles=excl if excl else None,
                D=D, verbose=True)
            # stamp η on the graph for downstream consumers
            self.G.graph['carved_sim_eta'] = float(eta)
            self.G.graph['carved_sim_D_areal'] = float(D)
            self.G.graph['carved_sim_wall_model'] = 'constant_loss_tangent'
            # Apply per-tile π-flip to carved sim fields (matches sync_v2
            # convention on measured; see _apply_pi_flip_and_persist_sim).
            self._apply_pi_flip_and_persist_sim(mode='carved')
            dt = time.time() - t0
            self._carved_status.appendPlainText(
                f"Done in {dt:.0f} s.\n"
                f"  v2 fields recomputed: {stats['n_v2']}\n"
                f"  outlets: {stats['n_outlets']}\n"
                f"  Σα (total): {stats['alpha_sum']:.3f} nL/s\n"
                f"  sim fields written: {stats['n_sim_fields']}")
            # Unlock the 'Simulated (carved)' radio in the View tab.
            self._refresh_carved_sim_radio_enable()
            self._carved_stat_lbl.setText(self._carved_sim_status_text())
        except Exception as e:
            import traceback
            self._carved_status.appendPlainText(
                f"FAILED: {e}\n{traceback.format_exc()}")
        finally:
            self._carved_run_btn.setEnabled(True)


    def _build_simulate_tab(self, layout):
        """Forward transmission-line solver controls.

        MVP scope: BCs come from the stored `bc_harmonics` on boundary
        nodes (no custom override yet).  D and f₀ are user-adjustable.
        Results are written to the in-memory graph under the
        `_sim_tmp_*` namespace and added to the View-tab field dropdown
        when present.  Never persisted to the gpickle.
        """
        from qtpy.QtWidgets import (
            QLabel, QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout,
            QRadioButton, QButtonGroup, QDoubleSpinBox, QCheckBox,
            QComboBox as _QComboBox)

        # ── Detect default f₀ from the graph ──
        self._sim_detected_f0 = self._detect_sim_f0_hz()

        # ── BC source ──
        bc_group = QGroupBox("Boundary conditions")
        bc_lay = QVBoxLayout(bc_group)
        # Sole BC mode (2026-07-02): two-pass Dirichlet-tie ΔP with
        # per-boundary normalized-Q-spectrum matching.
        #   Pass 1: Dirichlet ΔP at DC only, no AC drive.  Extract Q_0
        #           at every boundary from the DC solve.
        #   Pass 2: fully Neumann Q at every boundary; each Q at
        #           harmonic k is Q_0 · (bc_harmonics[k] / bc_harmonics[0]),
        #           i.e. Q_0 scaled by that node's measured normalized
        #           spectrum.
        # Result: Q_V and Q_A both reproduce the measured normalized
        # Q spectrum (|H_k/H_0| and phases) at every boundary while DC
        # is driven by ΔP.
        self._sim_bc_radio_dirichlet = QRadioButton(
            "Dirichlet-tie ΔP + normalized-Q match")
        # Not offered as a UI option in plexusflow — see note below where
        # the simplified BC scheme is described.  Always False so
        # _run_simulation takes the equal-flux branch.
        self._sim_bc_radio_dirichlet.setChecked(False)
        self._sim_bc_radio_dirichlet.hide()
        self._sim_bc_radio_dirichlet.setToolTip(
            "Sole simulation BC mode.\n\n"
            "Two-pass solve:\n"
            "  1) Pure DC Dirichlet: P_A − P_V = ΔP (below), no AC drive.\n"
            "     Extract Q̂_DC(n) at every boundary from the DC solve\n"
            "     (summed from edge_flows, respects conservation exactly).\n"
            "  2) Fully Neumann Q at every boundary:\n"
            "       target[0] = Q̂_DC(n)                (predicted DC)\n"
            "       target[k] = Q̂_DC(n) · bch[k]/bch[0]   (measured PI ratio)\n"
            "     The INJECTED PI at each boundary equals the MEASURED PI\n"
            "     bch[k]/bch[0], by construction — that's the boundary\n"
            "     invariant we trust.  AC magnitude scales with predicted\n"
            "     DC (so an AC-magnitude test is coupled to DC accuracy).\n\n"
            "Result: measured PI ratio preserved at every boundary; DC\n"
            "magnitude at every boundary is the geometry prediction.")
        # Legacy radios removed 2026-07-02 (were: classwave, measured,
        # custom).  Kept as attributes for back-compat with other code
        # paths that check .isChecked() — they're always False now.
        self._sim_bc_radio_classwave = QRadioButton()
        self._sim_bc_radio_classwave.setChecked(False)
        self._sim_bc_radio_classwave.hide()
        self._sim_bc_radio_measured = QRadioButton()
        self._sim_bc_radio_measured.setChecked(False)
        self._sim_bc_radio_measured.hide()
        self._sim_bc_radio_custom = QRadioButton()
        self._sim_bc_radio_custom.setChecked(False)
        self._sim_bc_radio_custom.hide()
        # NEW BC mode: equal-flux at A (Q(A1)=Q(A2)=Q_A_total/2), V grounded.
        # Trust the anatomical claim that paired A ports have equal DC flow;
        # let the total flux be a knob (default = measured A DC sum, but
        # user can scan it).  The A pressures P(A1), P(A2) become model
        # OUTPUTS — if geometry is self-consistent with A1=A2 they should
        # be near-equal; if not, the split gives a diagnostic of network
        # asymmetry.  V pressures pinned to 0 at all harmonics.  AC at A
        # uses measured PI ratio × prescribed Q_A/2 per port; V AC is
        # a full model output.
        self._sim_bc_radio_equalflux = QRadioButton(
            "Equal-flux at A + grounded V")
        # Simplified plexusflow BC: this is the SOLE mode.  Always True so
        # _run_simulation takes the two-pass equal-flux branch.
        self._sim_bc_radio_equalflux.setChecked(True)
        self._sim_bc_radio_equalflux.hide()
        self._sim_bc_radio_equalflux.setToolTip(
            "Q(A1) = Q(A2) = Q_A_total / 2, prescribed as Q-BC at both A\n"
            "nodes.  V grounded at P=0 (all harmonics).  AC at A uses\n"
            "measured PI × Q_A/2 per port (measured PI preserved).\n\n"
            "Q_A_total (spinbox below) defaults to the sum of measured A\n"
            "DC — sweep it to explore how V split responds to total flux.\n\n"
            "Diagnostic prints P(A1), P(A2), P(V1), P(V2) as solver outputs\n"
            "— non-equal A pressures indicate geometric asymmetry between\n"
            "the two arterial paths.")
        bc_group_buttons = QButtonGroup(bc_group)
        bc_group_buttons.addButton(self._sim_bc_radio_dirichlet)
        bc_group_buttons.addButton(self._sim_bc_radio_equalflux)
        # BC-mode radios hidden — plexusflow ships with a single BC scheme.
        # A static label documents what runs:
        _bc_scheme_lbl = QLabel(
            "<b>Sole BC scheme</b> — two-pass, mixed BC by harmonic:<br>"
            "&nbsp;&nbsp;<b>DC</b>: <tt>Q(A_r) = Q_A_total / N_A</tt> at every A (Neumann); "
            "<tt>P(V_r) = 0</tt> at every V (Dirichlet, common reservoir).<br>"
            "&nbsp;&nbsp;→ Pass 1 solves for the network-determined venous split "
            "<tt>Q&#770;_DC(V_r)</tt>.<br>"
            "&nbsp;&nbsp;<b>AC (n≥1)</b>: fully Neumann at every boundary — "
            "<tt>Q^(n)(A_r) = (Q_A_total/N_A) · bch[n]/bch[0]</tt>, "
            "<tt>Q^(n)(V_r) = Q&#770;_DC(V_r) · bch[n]/bch[0]</tt>.")
        _bc_scheme_lbl.setWordWrap(True)
        bc_lay.addWidget(_bc_scheme_lbl)

        # Q_A_total spinbox (used only in equal-flux mode)
        qa_row = QHBoxLayout()
        qa_row.addWidget(QLabel("Q_A_total (nL/s, prescribed arterial DC):"))
        self._sim_qa_spin = QDoubleSpinBox()
        self._sim_qa_spin.setDecimals(3)
        self._sim_qa_spin.setRange(0.001, 100.0)
        self._sim_qa_spin.setSingleStep(0.1)
        # Default = Σ measured A DC (bc_harmonics[0].real over source nodes).
        # Falls back to 1.0 nL/s if no A bc_harmonics available.
        qa_default = 0.0
        for _an in self._source_nodes:
            _bch = self.G.nodes[_an].get('bc_harmonics')
            if _bch is not None:
                try:
                    qa_default += float(np.asarray(_bch, dtype=complex)[0].real)
                except Exception:
                    pass
        if not (np.isfinite(qa_default) and qa_default > 0):
            qa_default = 1.0
        self._sim_qa_spin.setValue(qa_default)
        self._sim_qa_spin.setToolTip(
            "Total arterial DC flux (Σ over A ports).  In equal-flux mode,\n"
            "each A port is prescribed Q(Ai) = Q_A_total / N_A.\n\n"
            "Default = Σ measured A DC (from bc_harmonics on A nodes) so\n"
            "the sim's absolute magnitudes match the measurement.  Scan\n"
            "to see how the venous split and interior flows respond.")
        qa_row.addWidget(self._sim_qa_spin)
        qa_row.addStretch(1)
        bc_lay.addLayout(qa_row)

        # ── V-closure sub-selector ────────────────────────────────────
        # Only relevant in the Dirichlet-tie path.  Pass 1 always uses
        # common P_V = 0.  In Pass 2:
        #   flux    — Q-BC at every boundary including V (current default;
        #             matches historical behaviour but effectively pins
        #             each V node's DC to its Pass-1 prediction).
        #   pressure — Q-BC at A only; V held at Dirichlet P=0 across all
        #             harmonics (grounded venous).  V flows are then
        #             MODEL OUTPUTS, not BCs.  Cleaner if you want the
        #             venous DC split to remain a network prediction.
        v_closure_group = QGroupBox("V closure (Pass 2)")
        v_closure_lay = QHBoxLayout(v_closure_group)
        self._sim_v_closure_flux = QRadioButton("Per-node flux at V")
        # Sole V-closure in plexusflow — Pass 2 injects
        # Q^(n)(V_r) = Q̂_DC(V_r) × bch[n]/bch[0] (paper-outline scheme).
        self._sim_v_closure_flux.setChecked(True)
        self._sim_v_closure_flux.setToolTip(
            "Pass 2 applies a Q-BC at every boundary (A and V).\n"
            "V DC values equal Pass-1 predictions; V AC values are\n"
            "Q_pred × measured (H_k/H_0).  Effectively fully-Neumann.\n"
            "Uses measured V waveform as an input.")
        self._sim_v_closure_pressure = QRadioButton("Common pressure at V (grounded, P_V_AC=0)")
        self._sim_v_closure_pressure.setToolTip(
            "V nodes held at Dirichlet P=0 across ALL harmonics.\n"
            "Assumes venous reservoir has NO AC pressure fluctuation\n"
            "— a strong simplification (real embryonic atrium DOES\n"
            "oscillate). V flows fully predicted; measured V waveform\n"
            "becomes a test target, not an input.")
        # "Tied common P at V (nonzero AC)" — uses the solver's new
        # `tied_pressure_nodes` parameter, which adds a penalty
        # admittance between listed nodes (forces equal P at every
        # harmonic) and excludes them from Q-BC / conservation
        # rescaling.  V AC pressure becomes a solve output constrained
        # to be equal at both V ports (shared-atrium physics).
        self._sim_v_closure_meas_ac = QRadioButton(
            "Predicted DC + measured AC at V (raw phasor)")
        # Not offered in plexusflow — kept as attribute for _run_simulation
        # branch check; forced False.
        self._sim_v_closure_meas_ac.setChecked(False)
        self._sim_v_closure_meas_ac.setToolTip(
            "V DC = Q̂_DC(V) predicted from Pass 1 (untrusted V DC "
            "magnitude replaced by geometry).\n"
            "V AC target[k] = bch[k] MEASURED PHASOR DIRECTLY at every "
            "harmonic (k=1,2,3) — no PI rescaling to predicted DC.\n\n"
            "Use when: V DC magnitudes are untrustworthy but V AC "
            "amplitude AND phase are trustworthy (atrial pressure "
            "oscillation is heart-driven, doesn't scale with vessel "
            "cross-section bias).")
        self._sim_v_closure_tied = QRadioButton("Tied common P at V (nonzero AC)")
        self._sim_v_closure_tied.setToolTip(
            "Adds a penalty admittance between V nodes in the solver so\n"
            "P_V1(ω) ≈ P_V2(ω) at EVERY harmonic — including AC.  V nodes\n"
            "get no Q-BC (they're network outputs, not inputs).  V DC\n"
            "gauge is set by the interior pin.\n\n"
            "Physically: shared atrial chamber with oscillating pressure.\n"
            "V AC pressure develops from network response to A drive,\n"
            "but constrained to be equal at both V ports.\n"
            "V AC flow is a genuine model output — can be tested against\n"
            "measured V waveform.")
        _v_closure_bg = QButtonGroup(v_closure_group)
        _v_closure_bg.addButton(self._sim_v_closure_flux)
        _v_closure_bg.addButton(self._sim_v_closure_pressure)
        _v_closure_bg.addButton(self._sim_v_closure_meas_ac)
        _v_closure_bg.addButton(self._sim_v_closure_tied)
        # V-closure sub-selector hidden — plexusflow uses the "Per-node
        # flux at V" scheme (documented in the top BC-scheme label above).
        # The QRadioButton objects still exist so downstream .isChecked()
        # calls don't crash.  Layout add-widgets omitted below.
        v_closure_group.hide()
        v_closure_lay.addStretch(1)
        bc_lay.addWidget(v_closure_group)

        # DC-consistency diagnostic checkbox
        self._sim_dc_diag_chk = QCheckBox("Print Pass-1 vs measured DC per port")
        self._sim_dc_diag_chk.setChecked(True)
        self._sim_dc_diag_chk.setToolTip(
            "After Pass 1, log a per-port table comparing:\n"
            "  Q_pred (from ΔP + geometry + D)  vs\n"
            "  Q_meas = Re(bc_harmonics[0])     — the measured DC at that node.\n\n"
            "Their ratio and difference tell you how well the network\n"
            "geometry alone explains the measured DC split — separately\n"
            "for arterial and venous ports.  If either class has a big\n"
            "systematic mismatch, that's a geometry-model failure.")
        bc_lay.addWidget(self._sim_dc_diag_chk)

        # Quick summary of available boundary nodes.
        bc_info_lines = [
            f"Sources (A): {len(self._source_nodes)}",
            f"Sinks (V):   {len(self._sink_nodes)}",
        ]
        bc_lay.addWidget(QLabel("<small><tt>"
                                 + "<br>".join(bc_info_lines)
                                 + "</tt></small>"))

        # ── Custom waveform sub-panel ──
        # Twelve spinboxes: amp fraction (|H_k|/|Q̄|) + phase (degrees)
        # for H1, H2, H3 on EACH side (A and V) independently.  DC
        # magnitude is set automatically from the Inlet/outlet flux +
        # N_src / N_snk and the sign convention (+ on A, − on V into
        # the network).  Equal-split mode is the default — every A
        # boundary gets the same A waveform and every V boundary gets
        # the same V waveform.  Live preview reflects both at once.
        self._sim_custom_group = QGroupBox(
            "Custom waveform shape  (independent A and V)")
        cust_lay = QVBoxLayout(self._sim_custom_group)
        _cust_help_lbl = QLabel(
            "<small>Specify the A and V waveforms separately.  Each "
            "row gives the amp fraction (|H<sub>k</sub>|/|Q̄|) and "
            "phase (degrees) per harmonic for that side.  The chosen "
            "shape is applied to ALL boundaries of that type — every "
            "A gets the A spec, every V gets the V spec.  DC magnitudes "
            "come from flux / N.  Preview shows both with the sign "
            "chosen so DC reads positive on each side.</small>")
        _cust_help_lbl.setWordWrap(True)
        cust_lay.addWidget(_cust_help_lbl)
        # Twelve spinboxes — (side, harmonic, kind) → spinbox.
        # Header row labels the columns.
        hdr_row = QHBoxLayout()
        hdr_row.addWidget(QLabel("<b></b>"))
        hdr_row.addWidget(QLabel("<b>A amp</b>"))
        hdr_row.addWidget(QLabel("<b>A phase°</b>"))
        hdr_row.addWidget(QLabel("<b>V amp</b>"))
        hdr_row.addWidget(QLabel("<b>V phase°</b>"))
        hdr_row.addStretch(1)
        cust_lay.addLayout(hdr_row)
        self._sim_custom_spins = {}   # (side, harmonic, kind) → spinbox
        # Sensible defaults: H1 ≈ 0.5 (PI ~ 1), H2 ≈ 0.15, H3 ≈ 0.05.
        # Same defaults for both sides; the user changes V to taste.
        _default_amps = {1: 0.5, 2: 0.15, 3: 0.05}
        for k in (1, 2, 3):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"H{k}:"))
            for side in ('A', 'V'):
                amp = QDoubleSpinBox()
                amp.setDecimals(3); amp.setRange(0.0, 5.0)
                amp.setSingleStep(0.05)
                amp.setValue(_default_amps[k])
                amp.valueChanged.connect(self._refresh_custom_preview)
                row.addWidget(amp)
                phase = QDoubleSpinBox()
                phase.setDecimals(1); phase.setRange(-360.0, 360.0)
                phase.setSingleStep(5.0); phase.setValue(0.0)
                phase.valueChanged.connect(self._refresh_custom_preview)
                row.addWidget(phase)
                self._sim_custom_spins[(side, k, 'amp')] = amp
                self._sim_custom_spins[(side, k, 'phase')] = phase
            row.addStretch(1)
            cust_lay.addLayout(row)
        # Preview canvas (small matplotlib figure)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as _FigureCanvas)
        self._sim_custom_fig = Figure(figsize=(4.0, 1.7), dpi=90,
                                       tight_layout=True)
        self._sim_custom_canvas = _FigureCanvas(self._sim_custom_fig)
        self._sim_custom_canvas.setMinimumHeight(150)
        cust_lay.addWidget(self._sim_custom_canvas)
        # Initial enable state — spinboxes disabled (measured is the
        # default BC source), preview canvas always visible.
        for sp in self._sim_custom_spins.values():
            sp.setEnabled(False)
        # Hidden unless "Custom BCs" is selected — the class-waveform default
        # and measured mode don't need the spinbox grid + preview (saves a
        # large amount of vertical space).
        self._sim_custom_group.setVisible(False)
        bc_lay.addWidget(self._sim_custom_group)
        # Legacy custom-waveform editor — only relevant for the removed
        # Custom BC mode.  Hidden in the simplified plexusflow tab.
        self._sim_custom_group.hide()
        layout.addWidget(bc_group)
        # Draw the initial preview after widgets exist (will reflect
        # measured BCs since that's the default selection).  Connect
        # the flux + equal-split widgets later — those signals are
        # wired below once the corresponding controls are built.

        # ── Show BC waveforms ──
        # Pop a dialog with the measured Q(t) at every source and sink,
        # read straight from `bc_Q_t` on each boundary node.  Two panels:
        # all A's overlaid on the left, all V's overlaid on the right.
        bc_plot_btn = QPushButton("Show boundary Q(t) waveforms")
        bc_plot_btn.setToolTip(
            "Plot the measured Q(t) at every A source and every V sink,\n"
            "as stored in each boundary node's bc_Q_t.  Straight from\n"
            "the graph — no reconstruction, no simulation.")
        bc_plot_btn.clicked.connect(self._show_bc_waveforms)
        layout.addWidget(bc_plot_btn)

        # ── Parameters ──
        param_group = QGroupBox("Parameters")
        param_lay = QVBoxLayout(param_group)
        # D (distensibility) — areal convention.  Split into mantissa
        # and exponent so the user can pick orders of magnitude
        # explicitly: D = mantissa × 10^exponent.  Default 1.0e-3.
        d_row = QHBoxLayout()
        d_row.addWidget(QLabel("D (1/Pa, areal):"))
        self._sim_D_mantissa = QDoubleSpinBox()
        self._sim_D_mantissa.setDecimals(2)
        self._sim_D_mantissa.setRange(1.0, 9.99)
        self._sim_D_mantissa.setSingleStep(0.1)
        self._sim_D_mantissa.setValue(1.0)
        self._sim_D_mantissa.setToolTip(
            "Mantissa for D in scientific notation (1.00 to 9.99).")
        d_row.addWidget(self._sim_D_mantissa)
        d_row.addWidget(QLabel("× 10^"))
        from qtpy.QtWidgets import QSpinBox
        self._sim_D_exponent = QSpinBox()
        self._sim_D_exponent.setRange(-7, -1)
        self._sim_D_exponent.setSingleStep(1)
        self._sim_D_exponent.setValue(-3)
        self._sim_D_exponent.setToolTip(
            "Exponent for D in scientific notation (−7 to −1).\n"
            "Areal convention: typical HH-stage yolk-sac D ≈ 1.3e-3.")
        d_row.addWidget(self._sim_D_exponent)
        d_row.addStretch(1)
        param_lay.addLayout(d_row)
        # η — wall loss angle (constant-loss / structural-damping form).
        # D_eff(ω) = D · (1 − iη) applied per harmonic via a monkey-patch
        # of _vessel_admittance during the solve. η=0 → elastic.
        eta_row = QHBoxLayout()
        eta_row.addWidget(QLabel("η (loss tan, wall):"))
        self._sim_eta_spin = QDoubleSpinBox()
        self._sim_eta_spin.setDecimals(2)
        self._sim_eta_spin.setRange(0.0, 2.0)
        self._sim_eta_spin.setSingleStep(0.05)
        self._sim_eta_spin.setValue(0.0)
        self._sim_eta_spin.setToolTip(
            "Wall loss tangent η in D_eff(ω) = D·(1 − iη).\n"
            "0 = elastic (default). ~0.4 matches the current fit sweet spot.")
        eta_row.addWidget(self._sim_eta_spin)
        eta_row.addStretch(1)
        param_lay.addLayout(eta_row)
        # f0 (read-only label, auto-detected).
        f0_row = QHBoxLayout()
        f0_row.addWidget(QLabel("f₀ (Hz):"))
        f0_val = (f"{self._sim_detected_f0:.3f}"
                  if np.isfinite(self._sim_detected_f0) else "n/a")
        self._sim_f0_lbl = QLabel(f"<tt>{f0_val}</tt>  "
                                   "<small>(auto from graph)</small>")
        f0_row.addWidget(self._sim_f0_lbl)
        f0_row.addStretch(1)
        param_lay.addLayout(f0_row)
        # Inlet/outlet flux: scale BCs so total source DC = this value.
        flux_row = QHBoxLayout()
        flux_row.addWidget(QLabel("Inlet/outlet flux (nL/s):"))
        self._sim_flux_spin = QDoubleSpinBox()
        self._sim_flux_spin.setDecimals(3)
        self._sim_flux_spin.setRange(0.001, 1000.0)
        self._sim_flux_spin.setSingleStep(0.1)
        self._sim_flux_spin.setValue(1.0)
        self._sim_flux_spin.setToolTip(
            "Total inflow (Σ source DC) is scaled to this value.  All\n"
            "boundary harmonics are scaled by the same factor, so the\n"
            "AC content's relative shape is preserved.  The solver's\n"
            "conservation step then enforces Σ sinks = Σ sources.")
        flux_row.addWidget(self._sim_flux_spin)
        flux_row.addStretch(1)
        # Legacy — only used by the removed Dirichlet-tie BC mode.
        # Hidden but kept so downstream code can still read .value().
        for _i in range(flux_row.count()):
            w = flux_row.itemAt(_i).widget()
            if w is not None: w.hide()
        # Mean pressure difference ΔP = P_A − P_V (Pa).  Active only
        # when the Dirichlet-tie BC mode is selected.  Set to 0 to fall
        # back to flux-driven mode (V is Dirichlet-tied, A is Q-driven).
        # Non-zero → pure Dirichlet: P at A and V both prescribed,
        # arterial Q becomes a solve output.
        dp_row = QHBoxLayout()
        dp_row.addWidget(QLabel("Mean ΔP = P_A − P_V  (Pa):"))
        self._sim_dp_spin = QDoubleSpinBox()
        self._sim_dp_spin.setDecimals(2)
        self._sim_dp_spin.setRange(0.0, 1000.0)
        self._sim_dp_spin.setSingleStep(1.0)
        self._sim_dp_spin.setValue(10.0)          # default 10 Pa (2026-07-02)
        self._sim_dp_spin.setToolTip(
            "ΔP = P_A − P_V (Pa), applied at DC in Pass 1.\n"
            "Default 10 Pa.  Q_0 at every boundary comes out of that\n"
            "DC solve; Pass 2 then scales the measured normalized Q\n"
            "spectrum at each boundary by Q_0 to build Neumann Q BCs.")
        dp_row.addWidget(self._sim_dp_spin)
        dp_row.addStretch(1)
        # Legacy — only used by the removed Dirichlet-tie BC mode.  Hidden.
        for _i in range(dp_row.count()):
            w = dp_row.itemAt(_i).widget()
            if w is not None: w.hide()
        # V-side BC handling: lets the user switch from Q-BC at V
        # (default, measured Q_V drives the network) to pressure
        # matching at V, with two flavours.  Both P-match modes turn
        # the measured Q_V into a validation target rather than an
        # input — what the user trusts physically is constrained, what
        # they trust empirically can be checked independently.

        # DC distribution + per-type AC averaging:
        # - Unchecked: preserve each boundary's measured ratios.
        # - Checked: every A gets the SAME waveform (DC = flux/N_src
        #   plus the complex mean of measured AC across all A's); same
        #   for V (with sign flip).  The A↔V phase difference (e.g.
        #   cardiac antiphase) is preserved across types.
        self._sim_equal_split_chk = QCheckBox(
            "Uniform per-type BCs  "
            "(one waveform per A's, one per V's)")
        self._sim_equal_split_chk.setChecked(True)
        self._sim_equal_split_chk.setToolTip(
            "Checked (default):  every A gets the SAME waveform and\n"
            "every V gets the SAME waveform.  In Custom mode that's\n"
            "the A / V spec from the spinboxes; in Measured mode it's\n"
            "the per-type complex average of stored bc_harmonics.\n"
            "DC is exactly ±flux/N per side, so mass balance is exact\n"
            "by construction and the solver rebalance is a no-op.\n\n"
            "Unchecked:  each boundary keeps its measured per-node\n"
            "ratios (Measured mode only; meaningless for Custom\n"
            "because the Custom spec is already per-type).  The\n"
            "solver's DC-conservation step may rescale slightly so\n"
            "Σ = 0; applied total flux can drift from the target.")
        # Legacy — Dirichlet-tie-mode toggle, always effectively "per-type"
        # in the equal-flux BC scheme.  Hidden.
        self._sim_equal_split_chk.hide()
        # Live BC preview reacts to flux + equal-split changes too, so
        # the user always sees the actual scaled waveform.
        self._sim_flux_spin.valueChanged.connect(
            self._refresh_custom_preview)
        self._sim_equal_split_chk.toggled.connect(
            self._refresh_custom_preview)
        # Draw the initial preview now that all the inputs exist.
        self._refresh_custom_preview()
        # Equalize the inflow stem conductance per A node.  For each
        # arterial source, walk from A through degree-2 nodes to the
        # first branch point — that's the "inflow stem".  Override
        # every stem edge's radius so each A's stem has the same total
        # Poiseuille resistance (median across A's).  Lengths are kept;
        # only radii change.  Useful when stem lengths differ
        # (e.g. one A has a single edge to first branch, another has
        # two edges) and you want symmetric inflow pressure drops.
        # Equalize A-stem conductance — UI removed for now.  The checkbox
        # object stays parentless (so `.isChecked()` still works in the
        # solver path and returns False), but isn't added to any layout.
        # To restore: re-add `param_lay.addWidget(self._sim_equalize_A_chk)`.
        self._sim_equalize_A_chk = QCheckBox(
            "Equalize A-stem conductance  (A → first-branch point)")
        self._sim_equalize_A_chk.setChecked(False)
        # Geometry override mode: measured, uniform radius only, or
        # uniform conductance (R AND L constant).  All three keep the
        # original graph intact; only the solve-time copy is altered.
        geom_row = QHBoxLayout()
        geom_row.addWidget(QLabel("Geometry:"))
        self._sim_geom_combo = _QComboBox()
        self._sim_geom_combo.addItems([
            "Measured (default)",
            "Uniform radius  (R const, L from measured)",
            "Uniform conductance  (R, L both constant)",
        ])
        self._sim_geom_combo.setToolTip(
            "Measured: use the graph's stored radius/length per edge.\n"
            "Uniform radius: override every R to 5 px; conductance is\n"
            "  then determined by per-edge length alone.\n"
            "Uniform conductance: override both R (5 px) and L (50 px),\n"
            "  so flow distribution is purely topological.\n"
            "Original graph is unchanged; only the solve sees overrides.")
        geom_row.addWidget(self._sim_geom_combo)
        geom_row.addStretch(1)
        param_lay.addLayout(geom_row)
        layout.addWidget(param_group)

        # ── Actions ──
        actions_row = QHBoxLayout()
        self._sim_run_btn = QPushButton("Run simulation")
        self._sim_run_btn.clicked.connect(self._run_simulation)
        actions_row.addWidget(self._sim_run_btn)
        self._sim_clear_btn = QPushButton("Clear simulation")
        self._sim_clear_btn.setEnabled(False)
        self._sim_clear_btn.clicked.connect(self._clear_simulation)
        actions_row.addWidget(self._sim_clear_btn)
        self._sim_save_btn = QPushButton("Save graph state…")
        self._sim_save_btn.setToolTip(
            "Pickle the current in-memory graph (raw + normalized +\n"
            "sim + tile_gauges + all metadata) to a *.gpickle.  Reload\n"
            "the viewer against that file and everything — normalized\n"
            "fields, sim results, and the UI parameter selections — is\n"
            "restored automatically.")
        self._sim_save_btn.clicked.connect(self._on_save_state_clicked)
        actions_row.addWidget(self._sim_save_btn)
        layout.addLayout(actions_row)

        # ── Status ──
        self._sim_status_lbl = QLabel(
            "<small><i>No simulation has been run yet.  Results live "
            "in memory under `_sim_tmp_*` fields and disappear when "
            "you close the viewer.</i></small>")
        self._sim_status_lbl.setWordWrap(True)
        layout.addWidget(self._sim_status_lbl)

        # ── Loaded-sim indicator (populated from G.graph['sim_meta']
        #    if the graph was saved after a previous run) ──
        self._sim_loaded_lbl = QLabel("")
        self._sim_loaded_lbl.setWordWrap(True)
        self._sim_loaded_lbl.setStyleSheet(
            "QLabel { color: #446; background: #eef4ff; "
            "padding: 4px; border: 1px solid #99b; }")
        self._sim_loaded_lbl.hide()
        layout.addWidget(self._sim_loaded_lbl)
        layout.addStretch(1)

        # ── No-boundary guard ──
        # If the graph carries no source / sink designations (e.g. a
        # generic mosaic without A/V assignment), every control on
        # this tab is meaningless — the solver has nothing to drive
        # the flow with.  Disable the whole tab's widgets and show
        # a single explanatory label instead of half-working spinboxes.
        if not self._source_nodes and not self._sink_nodes:
            for w in (bc_group, param_group, actions_row,
                      self._sim_status_lbl):
                _w = (w if hasattr(w, 'setEnabled')
                       else None)  # actions_row is a layout
                if _w is not None:
                    _w.setEnabled(False)
            # Disable layout-only children too (actions_row).
            for i in range(actions_row.count()):
                item = actions_row.itemAt(i).widget()
                if item is not None:
                    item.setEnabled(False)
            # Replace the bottom status with a clear message.
            self._sim_status_lbl.setText(
                "<small><b>Simulation disabled.</b>  This graph has no "
                "boundary nodes designated as sources (A) or sinks (V), "
                "so there's nothing to drive a flow simulation.  Add "
                "<code>boundary_type</code> annotations (\"source\" / "
                "\"sink\") to the relevant nodes in the gpickle to "
                "enable this tab.</small>")
            self._sim_status_lbl.setEnabled(True)  # leave the note readable

        # If the graph carries a saved sim_meta from a prior session,
        # restore the UI controls and show the "loaded" banner.
        try:
            self._restore_sim_ui_from_graph()
        except Exception as _exc:
            print(f"[sim] restore from graph failed: {_exc}")


    # ══════════════════════════════════════════════════════════════════
    # Kinematic pump tab — reduced load-coupled tubular-heart model
    # (reuses scripts/sim_kinematic_load_pump.py)
    # ══════════════════════════════════════════════════════════════════

    def _build_pump_tab(self, layout):
        """Kinematic tubular-heart pump: prescribed heart motion, arterial
        (Q_out) / venous (Q_in) waveforms respond to vascular load (ligation).
        Defaults are the Somites21 DA/SV-calibrated pump.  Visualises the
        flow + pressure waveforms; writes per-port pump flux to `_pump_*`
        node fields."""
        from qtpy.QtWidgets import (
            QLabel, QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout,
            QDoubleSpinBox, QSpinBox, QComboBox as _QComboBox)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as _FigureCanvas)

        intro = QLabel(
            "<b>Kinematic tubular-heart pump</b> — the heart MOTION is "
            "prescribed; arterial ejection (Q_out) and venous filling (Q_in) "
            "are solved and RESPOND to vascular load.  Defaults = the "
            "Somites21 DA/SV-calibrated pump.  Ligate / partially occlude a "
            "boundary port to see the load response.")
        intro.setWordWrap(True); layout.addWidget(intro)

        self._pump_spins = {}
        pg = QGroupBox("Pump parameters"); pl = QVBoxLayout(pg)

        def add_param(key, label, lo, hi, step, val):
            row = QHBoxLayout(); row.addWidget(QLabel(label))
            spn = QDoubleSpinBox(); spn.setDecimals(2); spn.setRange(lo, hi)
            spn.setSingleStep(step); spn.setValue(val)
            row.addWidget(spn); row.addStretch(1); pl.addLayout(row)
            self._pump_spins[key] = spn

        add_param("closure_ratio", "Closure ratio (stroke):", 0.2, 0.95, 0.05, 0.95)
        add_param("duty", "Duty cycle (active frac):", 0.2, 0.9, 0.05, 0.35)
        add_param("tau_A", "τ_A  arterial Windkessel:", 0.0, 1.8, 0.1, 1.26)
        add_param("tau_V", "τ_V  venous Windkessel:", 0.0, 1.0, 0.05, 0.34)
        add_param("C_H", "C_H  chamber compliance:", 0.05, 0.6, 0.02, 0.13)
        add_param("C_AV", "C_A=C_V  compartment compl.:", 0.05, 0.6, 0.02, 0.18)
        add_param("leak_eps", "Leak ε (backflow frac):", 0.0, 0.35, 0.02, 0.08)
        add_param("G_net0", "G_net0  (load scale):", 0.1, 1.0, 0.05, 0.25)
        layout.addWidget(pg)

        lg = QGroupBox("Ligation / occlusion"); ll = QVBoxLayout(lg)
        lr = QHBoxLayout(); lr.addWidget(QLabel("Ligate port:"))
        self._pump_lig_combo = _QComboBox()
        self._pump_lig_combo.addItem("None", userData=None)
        for n in (self._source_nodes + self._sink_nodes):
            bt = self.G.nodes[n].get("boundary_type", "?")
            self._pump_lig_combo.addItem(f"Node {n} ({bt})", userData=int(n))
        lr.addWidget(self._pump_lig_combo); lr.addStretch(1); ll.addLayout(lr)
        orow = QHBoxLayout(); orow.addWidget(QLabel("Occlusion %:"))
        self._pump_occl = QSpinBox(); self._pump_occl.setRange(0, 100)
        self._pump_occl.setSingleStep(10); self._pump_occl.setValue(100)
        self._pump_occl.setToolTip("100% = full ligation; <100% = graded stenosis")
        orow.addWidget(self._pump_occl); orow.addStretch(1); ll.addLayout(orow)
        layout.addWidget(lg)

        self._pump_run_btn = QPushButton("Run pump")
        self._pump_run_btn.clicked.connect(self._run_pump)
        layout.addWidget(self._pump_run_btn)
        self._pump_status = QLabel(
            "<small><i>Not run yet.  The first run builds the DC load model "
            "(~20–40 s); later runs are fast.</i></small>")
        self._pump_status.setWordWrap(True); layout.addWidget(self._pump_status)

        self._pump_fig = Figure(figsize=(5.2, 4.6), dpi=90, tight_layout=True)
        self._pump_canvas = _FigureCanvas(self._pump_fig)
        self._pump_canvas.setMinimumHeight(380)
        layout.addWidget(self._pump_canvas)

        self._pump_state = None
        if not self._source_nodes or not self._sink_nodes:
            self._pump_run_btn.setEnabled(False)
            self._pump_status.setText(
                "<small><font color='#c33'>This graph has no A/V boundary "
                "nodes with bc_harmonics — the pump needs measured boundary "
                "waveforms to calibrate against.</font></small>")


    def _run_pump(self):
        from qtpy.QtWidgets import QApplication
        import sys as _sys
        from pathlib import Path as _Path
        _scripts = str(_Path(__file__).resolve().parents[2] / "scripts")
        if _scripts not in _sys.path:
            _sys.path.insert(0, _scripts)
        try:
            import sim_kinematic_load_pump as klp
            import sim_compartment as sc
        except Exception as e:
            self._pump_status.setText(
                f"<small><font color='#c33'>import failed: {e}</font></small>")
            return
        self._pump_run_btn.setEnabled(False)
        try:
            if self._pump_state is None:
                self._pump_status.setText("<small>Building DC load model "
                                          "(first run, ~20–40 s)…</small>")
                QApplication.processEvents()
                self._pump_state = sc.setup(str(self.graph_path), n_harmonics=3)
            state = self._pump_state
            p = klp.base_fixed_params(state["F0_HZ"])
            for k in ("closure_ratio", "duty", "tau_A", "tau_V", "C_H",
                      "C_AV", "leak_eps", "G_net0"):
                klp._set_param(p, k, float(self._pump_spins[k].value()))
            kin = klp.precompute_kinematics(p)
            bA = [int(b) for b in state["bidx_A"]]
            bV = [int(b) for b in state["bidx_V"]]
            Q0 = state["Qmeas_by_n"][0]
            A_DC = sum(abs(Q0[b]) for b in bA) * 1e12
            base = klp.graph_load(state, None); R0 = base["R_eff"]
            rec0 = klp.solve_pump(kin, p, p["G_net0"], n_cycles=60)
            scale = A_DC / max(rec0["Q_out"].mean(), klp.EPS)

            lp = self._pump_lig_combo.currentData()
            occl = self._pump_occl.value() / 100.0
            rec_lig = None; ktag = ""
            if lp is not None and occl > 0:
                if occl >= 1.0:
                    Rl = klp.graph_load(state, lp)["R_eff"]
                else:
                    Rl = klp.graded_occlusion_keff(state, lp, 1.0 - occl)
                kk = R0 / Rl
                rec_lig = klp.solve_pump(kin, p, p["G_net0"] * kk, n_cycles=60)
                ktag = f"  (ligation k={kk:.2f})"

            # ── Feed the pump's boundary waveforms into the LINEAR network
            # forward solve so ALL per-edge fields get recorded (DC, H1, H2, H3,
            # phase, pressure, PI).  Valid by harmonic balance: the
            # nonlinearity lives in the pump; the plexus is linear, so
            # per-harmonic propagation is exact.  (Interior AC is a forward
            # prediction — the pump's load model is DC-only — but there is no
            # interior measurement to contradict it.)
            # Feed the WINDKESSEL-FILTERED (DA/SV-equivalent) boundary waveforms
            # — the plexus boundary is the dorsal aorta, downstream of the
            # arterial Windkessel — not the raw ventricular pulse.
            r_use = rec_lig if rec_lig is not None else rec0
            Hout = np.asarray(klp._arch_filter(klp.to_harmonics(r_use["Q_out"]),
                                               p["tau_A"]), dtype=complex)
            Hin = np.asarray(klp._arch_filter(klp.to_harmonics(r_use["Q_in"]),
                                              p["tau_V"]), dtype=complex)
            bc_override = {}
            for b in bA:                         # sources: + into-network
                nid = state["nodes"][b]
                bc_override[nid] = Hout * (base["w_A"].get(b, 0.0) * scale)
            for b in bV:                         # sinks: − into-network
                nid = state["nodes"][b]
                bc_override[nid] = -Hin * (base["w_V"].get(b, 0.0) * scale)
            if lp is not None and occl >= 1.0 and lp in bc_override:
                bc_override[lp] = np.zeros(4, dtype=complex)   # full ligation: Q=0
            D_phys = 1.5e-3
            from plexusmech.tl.transmission_line import solve_transmission_line  # ported 2026-09-30
            result = solve_transmission_line(
                self.G, D=D_phys, n_harmonics=3, f0_hz=float(state["F0_HZ"]),
                bc_harmonics_override=bc_override, verbose=False)
            self._store_sim_result(result, float(state["F0_HZ"]), D_phys)

            self._plot_pump(rec0, rec_lig, scale, state["F0_HZ"], ktag)

            import numpy as _np
            def met(rec):
                Qo = rec["Q_out"]; Ho = klp.to_harmonics(Qo)
                return (Qo.mean() * scale, rec["P_A"].max(),
                        abs(Ho[1]) / max(abs(Ho[0]), klp.EPS))
            mo, af, pu = met(rec0)
            txt = (f"<small>baseline: output <b>{mo:.2f}</b> nL/s, afterload "
                   f"{af:.2f}, pulsatility {pu:.2f}{ktag}")
            if rec_lig is not None:
                ml, afl, pul = met(rec_lig)
                txt += (f"<br>→ ligated: output <b>{ml:.2f}</b> "
                        f"({(ml-mo)/max(mo,1e-9)*100:+.0f}%), afterload {afl:.2f}, "
                        f"pulsatility {pul:.2f}")
            txt += ("<br><i>All per-edge fields recorded — switch the View-tab "
                    "Source to <b>Simulated</b> to color the mosaic.</i></small>")
            self._pump_status.setText(txt)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._pump_status.setText(
                f"<small><font color='#c33'>pump run failed: {e}</font></small>")
        finally:
            self._pump_run_btn.setEnabled(True)


    def _plot_pump(self, rec0, rec_lig, scale, F0, ktag):
        T_ms = 1000.0 / F0; tau = rec0["tau"]
        fig = self._pump_fig; fig.clear()
        ax1 = fig.add_subplot(2, 1, 1); ax2 = fig.add_subplot(2, 1, 2)
        ax1.plot(tau*T_ms, rec0["Q_out"]*scale, "r", lw=2, label="Q_out (arterial)")
        ax1.plot(tau*T_ms, rec0["Q_in"]*scale, "b", lw=2, label="Q_in (venous)")
        if rec_lig is not None:
            ax1.plot(tau*T_ms, rec_lig["Q_out"]*scale, "r--", lw=1.3, label="Q_out lig")
            ax1.plot(tau*T_ms, rec_lig["Q_in"]*scale, "b--", lw=1.3, label="Q_in lig")
        ax1.axhline(0, color="grey", lw=0.5); ax1.legend(fontsize=7)
        ax1.set_ylabel("flow [nL/s]")
        ax1.set_title("Pump flows" + ktag, fontsize=9)
        ax2.plot(tau*T_ms, rec0["P_A"], "r", lw=1.3, label="P_A (art.)")
        ax2.plot(tau*T_ms, rec0["P_V"], "b", lw=1.3, label="P_V (ven.)")
        ax2.plot(tau*T_ms, rec0["P_H"], "k", lw=1.0, label="P_H (chamber)")
        ax2.axhline(0, color="grey", lw=0.5); ax2.legend(fontsize=7)
        ax2.set_xlabel("t [ms]"); ax2.set_ylabel("pressure (a.u.)")
        ax2.set_title("Compartment pressures", fontsize=9)
        self._pump_canvas.draw()


    # ══════════════════════════════════════════════════════════════════
    # End of inference adapter region.
    # ══════════════════════════════════════════════════════════════════

    def _show_bc_waveforms(self):
        """Pop a matplotlib dialog with two panels: all A (source) bc_Q_t
        overlaid on the left, all V (sink) bc_Q_t overlaid on the right.
        Reads straight from each boundary node's `bc_Q_t` — no phasor
        reconstruction, no simulation."""
        from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as _FigCanvas,
            NavigationToolbar2QT as _NavToolbar)
        FS = 250.0
        src_nodes = [n for n in self._source_nodes
                     if self.G.nodes[n].get('bc_Q_t') is not None]
        snk_nodes = [n for n in self._sink_nodes
                     if self.G.nodes[n].get('bc_Q_t') is not None]
        if not src_nodes and not snk_nodes:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.information(
                self.viewer.window._qt_window if self.viewer else None,
                "No BC waveforms",
                "No boundary nodes carry bc_Q_t.  Run a batch analyze "
                "that populates the recursive-KCL BC harmonics first.")
            return
        fig = Figure(figsize=(11, 4.4), tight_layout=True)
        for i, (nodes, title, colour) in enumerate([
                (src_nodes, "Sources (A) — bc_Q_t", "C0"),
                (snk_nodes, "Sinks (V) — bc_Q_t",   "C3")]):
            ax = fig.add_subplot(1, 2, i + 1)
            for n in nodes:
                qt = np.asarray(self.G.nodes[n]['bc_Q_t'], dtype=float)
                t = np.arange(len(qt)) / FS
                mean = float(np.mean(qt))
                ax.plot(t, qt, colour + '-', lw=1.5, alpha=0.85,
                        label=f"node {n}  ⟨Q⟩ = {mean:+.3f} nL/s")
            ax.axhline(0, color='k', lw=0.5, alpha=0.3)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("t (s)"); ax.set_ylabel("Q (nL/s)")
            ax.grid(alpha=0.3)
            if nodes:
                ax.legend(fontsize=8, loc='upper right')
        dlg = QDialog()
        dlg.setWindowTitle("Boundary Q(t) waveforms  (from bc_Q_t)")
        dlg.resize(1050, 480)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(
            "<i>Measured Q(t) at every boundary node — straight from "
            "<tt>bc_Q_t</tt>.  Not a sim output.</i>"))
        canvas = _FigCanvas(fig); v.addWidget(canvas)
        v.addWidget(_NavToolbar(canvas, dlg))
        dlg.exec_()


    def _detect_sim_f0_hz(self) -> float:
        """Pick the solver's default heart rate.  Priority:
        1. `G.graph['tile_f0_piv'][14]` — tile 14 is the BC / phase gauge
           (every bc_harmonics phasor is expressed at this f0).
        2. `G.graph['f0_canonical_hz']` — the normalization-plateau f0.
        3. `G.graph['sim_f0_hz']` if set by the pipeline.
        4. Median of `G.graph['tile_f0_piv']` (fallback).
        5. Median of `G.graph['tile_f0s']` (legacy kymograph).
        6. Constant 2.5 Hz.
        """
        tile_f0 = self.G.graph.get('tile_f0_piv') or {}
        for k in (14, '14'):
            v = tile_f0.get(k)
            if v is not None:
                try:
                    v = float(v)
                    if np.isfinite(v) and v > 0:
                        return v
                except (TypeError, ValueError):
                    pass
        f0_can = self.G.graph.get('f0_canonical_hz')
        if f0_can is not None:
            try:
                f0_can = float(f0_can)
                if np.isfinite(f0_can) and f0_can > 0:
                    return f0_can
            except (TypeError, ValueError):
                pass
        f0 = self.G.graph.get('sim_f0_hz')
        if f0 is not None:
            try:
                f0 = float(f0)
                if np.isfinite(f0) and f0 > 0:
                    return f0
            except (TypeError, ValueError):
                pass
        for key in ('tile_f0_piv', 'tile_f0s'):
            d = self.G.graph.get(key) or {}
            vals = []
            for v in d.values():
                try:
                    vv = float(v)
                    if np.isfinite(vv) and vv > 1.0:
                        vals.append(vv)
                except (TypeError, ValueError):
                    pass
            if vals:
                return float(np.median(vals))
        return 2.5


    def _find_inflow_stem(self, a_node) -> List[Tuple[int, int]]:
        """Walk from a boundary node through degree-2 nodes to the
        first node whose degree ≠ 2 (branch point or dead end).
        Returns the ordered list of edges traversed."""
        edges: List[Tuple[int, int]] = []
        prev = None
        curr = a_node
        while True:
            if curr != a_node and self.G.degree(curr) > 2:
                break  # branch point
            nbrs = [n for n in self.G.neighbors(curr) if n != prev]
            if not nbrs:
                break  # dead end
            nxt = nbrs[0]
            edges.append((curr, nxt))
            prev = curr
            curr = nxt
            # Cap to avoid pathological infinite loops on self-loops.
            if len(edges) > 64:
                break
        return edges


    def _stem_poiseuille_resistance(self, stem_edges) -> float:
        """Sum Σ 8μ L_e / (π R_e⁴) along a stem (Pa·s/m³)."""
        from ..analyze.constants import PX_SIZE_UM
        from ..analyze.constants import MU_DEFAULT
        px_to_m = PX_SIZE_UM * 1e-6
        total = 0.0
        for (u, v) in stem_edges:
            d = self.G.edges[u, v]
            try:
                R_px = float(d.get('radius'))
                L_px = float(d.get('length'))
            except (TypeError, ValueError):
                continue
            if not (R_px > 0 and L_px > 0):
                continue
            R_m = R_px * px_to_m
            L_m = L_px * px_to_m
            total += 8.0 * MU_DEFAULT * L_m / (np.pi * R_m ** 4)
        return total


    def _classwave_bc_override(self, bc_signed):
        """Phenomenological class-waveform boundary condition.

        For each class (sources=A, sinks=V), build per-port BC phasors
            Q_i^(0) = measured DC mean   (defines the share w_i)
            Q_i^(n) = w_i^(0) · A_c^(n)   for n ≥ 1
        where w_i = |Q_i^(0)| / Σ_class |Q_j^(0)| and A_c^(n) is the
        weighted least-squares class amplitude
            A_c^(n) = Σ_i w_i Q_i^(n) / Σ_i w_i² .

        `bc_signed`: node → complex [DC, H1, H2, H3] in into-network
        convention (sinks already sign-flipped).  Returns the same shape.
        This is the boundary-only model adopted June 2026 (see
        scripts/sim_compartment.py); the network solves the interior but
        interior AC is treated as indicative only.
        """
        out: Dict[int, np.ndarray] = {}
        for members_all in (self._source_nodes, self._sink_nodes):
            members = [n for n in members_all if n in bc_signed]
            if not members:
                continue
            dc = np.array([abs(complex(bc_signed[n][0])) for n in members])
            tot = float(dc.sum())
            w = dc / tot if tot > 0 else np.ones(len(members)) / len(members)
            for n in members:
                out[n] = np.asarray(bc_signed[n], dtype=complex).copy()
            n_harm = min(len(bc_signed[n]) for n in members)
            for h in range(1, n_harm):
                q = np.array([complex(bc_signed[n][h]) for n in members])
                A_c = np.sum(w * q) / max(float(np.sum(w ** 2)), 1e-30)
                for k, n in enumerate(members):
                    out[n][h] = w[k] * A_c
        return out


    def _run_simulation(self):
        """Forward transmission-line solve with a prescribed-Q boundary.

        Boundary phasors come from the selected BC source:
          • Class-waveform (default): Q_i^(n) = w_i^(0)·A_c^(n) — each port's
            DC-flow share times its class waveform, fit from bc_harmonics.
          • Measured: per-node bc_harmonics as-is.
          • Custom: user-specified per-class waveform.
        Results are written to the in-memory `_sim_tmp_*` fields.
        """
        if not self._source_nodes and not self._sink_nodes:
            self._sim_status_lbl.setText(
                "<small><font color='#c33'>No boundary nodes on this "
                "graph (need `boundary_type` set).  Cannot run.</font>"
                "</small>")
            return
        # Boundary-condition mode.
        dirichlet_tie = bool(getattr(self, '_sim_bc_radio_dirichlet', None)
                              is not None
                              and self._sim_bc_radio_dirichlet.isChecked())
        equal_flux_A = bool(getattr(self, '_sim_bc_radio_equalflux', None)
                             is not None
                             and self._sim_bc_radio_equalflux.isChecked())
        classwave = bool(getattr(self, '_sim_bc_radio_classwave', None)
                         is not None
                         and self._sim_bc_radio_classwave.isChecked())
        # Else: legacy prescribed-Q path (unchanged) ----------------------
        import time
        from qtpy.QtWidgets import QApplication
        D = (float(self._sim_D_mantissa.value())
             * (10.0 ** int(self._sim_D_exponent.value())))
        eta = float(getattr(self, '_sim_eta_spin', None).value()
                     if getattr(self, '_sim_eta_spin', None) is not None else 0.0)
        # Install a monkey-patch on _vessel_admittance so every solver call
        # made in this run sees D_eff(ω) = D · (1 − iη). Uses a class-level
        # pristine reference so repeated runs don't double-wrap.
        # Matches lock_D6p5e4_eta0p4.py's convention.
        from plexusmech.tl import transmission_line as _tl
        if not hasattr(type(self), '_pristine_vessel_admittance'):
            type(self)._pristine_vessel_admittance = _tl._vessel_admittance
        _orig_va = type(self)._pristine_vessel_admittance
        _eta_val = eta
        def _patched_vessel_admittance(R_m, L_m, omega, mu=_tl.MU_DEFAULT,
                                         rho=_tl.RHO_BLOOD, D=None):
            D_eff = (D if D is not None else 1e-6) * (1.0 - 1j * _eta_val)
            return _orig_va(R_m, L_m, omega, mu, rho, D_eff)
        _tl._vessel_admittance = _patched_vessel_admittance
        self._sim_last_eta = _eta_val
        # Re-detect f0 on every run so mid-session changes to
        # G.graph['f0_canonical_hz'] (e.g. after a fresh Normalize) are
        # picked up.  Also refresh the cached value used by the label.
        f0 = self._detect_sim_f0_hz()
        self._sim_detected_f0 = f0
        if not np.isfinite(f0):
            f0 = 2.5
        target_flux = float(self._sim_flux_spin.value())
        # Geometry override mode — combo index ⇒ behaviour:
        #   0 → measured            (no override)
        #   1 → uniform radius      (R const, L preserved)
        #   2 → uniform conductance (R AND L const)
        geom_mode_idx = int(self._sim_geom_combo.currentIndex())
        override_R = geom_mode_idx in (1, 2)
        override_L = geom_mode_idx == 2
        geom_tag = (
            "" if geom_mode_idx == 0
            else ", uniform R" if geom_mode_idx == 1
            else ", uniform R+L")
        split_tag = (", equal-split" if self._sim_equal_split_chk.isChecked()
                     else "")
        eqA_tag = (", A-stem eq" if self._sim_equalize_A_chk.isChecked()
                   else "")
        split_tag += eqA_tag
        v_bc_tag = ""
        if classwave:
            bc_src_tag = ", class-waveform BCs"
        elif self._sim_bc_radio_custom.isChecked():
            bc_src_tag = ", CUSTOM BCs"
        else:
            bc_src_tag = ""
        self._sim_run_btn.setEnabled(False)
        self._sim_status_lbl.setText(
            f"<small>Running solve at D = {D:.3g}, f₀ = {f0:.3f} Hz "
            f"flux = {target_flux:.3g} nL/s"
            f"{bc_src_tag}{geom_tag}{split_tag}{v_bc_tag} …</small>")
        QApplication.processEvents()

        # Build bc_harmonics_override.  Stored `bc_harmonics` on this
        # graph carries +outflow at sinks; the solver's "Q into network"
        # convention is +inflow everywhere, so we flip sinks by −1.
        # Two distribution modes:
        #   measured-ratio (default): scale all by `target_flux / |Σ src
        #     DC|` so the relative per-node imbalance is preserved.  The
        #     solver's DC-conservation step may then rescale slightly.
        #   equal-split: set DC = ±target_flux / N per type, keep AC
        #     harmonics measured.  Mass balance exact → no rebalance.
        equal_split = bool(self._sim_equal_split_chk.isChecked())
        custom_bc = bool(self._sim_bc_radio_custom.isChecked())
        bc_signed: Dict[int, np.ndarray] = {}
        sum_src_dc = 0.0
        for n in (self._source_nodes + self._sink_nodes):
            bch = self.G.nodes[n].get('bc_harmonics')
            if bch is None:
                continue
            s = -1.0 if n in self._sink_nodes else 1.0
            arr = s * np.asarray(bch, dtype=complex)
            bc_signed[n] = arr
            if n in self._source_nodes and np.isfinite(arr[0].real):
                sum_src_dc += float(arr[0].real)

        bc_override: Dict[int, np.ndarray] = {}
        if classwave:
            # Phenomenological boundary model: replace each port's measured
            # AC phasors with w_i·A_c (its DC-flow share × the class
            # waveform).  Measured DC is preserved (it defines w_i).
            bc_override = self._classwave_bc_override(bc_signed)
        elif custom_bc:
            # Build BCs from the user-specified waveforms.  A and V
            # specs are now independent — each side has its own amps +
            # phases.  Every A boundary gets the A spec, every V
            # boundary gets the V spec (with sign convention applied so
            # V's Q-into-network is negative-DC).
            (amps_A, phases_A), (amps_V, phases_V) = \
                self._read_custom_waveform()
            n_src = max(1, sum(1 for n in self._source_nodes
                               if n in bc_signed
                               or self.G.nodes[n].get('bc_harmonics')
                                   is not None))
            n_snk = max(1, sum(1 for n in self._sink_nodes
                               if n in bc_signed
                               or self.G.nodes[n].get('bc_harmonics')
                                   is not None))
            Qbar_A = +target_flux / n_src
            Qbar_V = -target_flux / n_snk  # canonical sign flip
            ac_A = [amps_A[i] * Qbar_A * np.exp(1j * phases_A[i])
                    for i in range(3)]
            ac_V = [amps_V[i] * Qbar_V * np.exp(1j * phases_V[i])
                    for i in range(3)]
            A_arr = np.array([Qbar_A] + ac_A, dtype=complex)
            V_arr = np.array([Qbar_V] + ac_V, dtype=complex)
            for n in self._source_nodes:
                bc_override[n] = A_arr.copy()
            for n in self._sink_nodes:
                bc_override[n] = V_arr.copy()
        elif equal_split:
            # Each boundary inside a type (A or V) gets THE SAME
            # waveform.  AC harmonics are the per-type complex average
            # of the measured BCs, so the A↔V phase difference (e.g.
            # ventricular ejection vs atrial suction) is preserved
            # across types but per-node spread within a type is gone.
            src_arrs = [bc_signed[n] for n in self._source_nodes
                        if n in bc_signed]
            snk_arrs = [bc_signed[n] for n in self._sink_nodes
                        if n in bc_signed]
            n_src = max(1, len(src_arrs))
            n_snk = max(1, len(snk_arrs))
            src_avg = (np.mean(np.stack(src_arrs, axis=0), axis=0)
                       if src_arrs else None)
            snk_avg = (np.mean(np.stack(snk_arrs, axis=0), axis=0)
                       if snk_arrs else None)
            if src_avg is not None:
                src_avg = src_avg.copy()
                src_avg[0] = +target_flux / n_src
            if snk_avg is not None:
                snk_avg = snk_avg.copy()
                snk_avg[0] = -target_flux / n_snk
            for n in bc_signed:
                if n in self._source_nodes and src_avg is not None:
                    bc_override[n] = src_avg.copy()
                elif n in self._sink_nodes and snk_avg is not None:
                    bc_override[n] = snk_avg.copy()
        else:
            if abs(sum_src_dc) > 1e-12:
                flux_scale = target_flux / abs(sum_src_dc)
            else:
                flux_scale = 1.0
            for n, arr in bc_signed.items():
                bc_override[n] = flux_scale * arr

        # Geometry overrides: save the keys we touch, override for the
        # duration of the solve, restore in `finally` so the mutation
        # is invisible to the rest of the viewer state.
        R_UNIFORM_PX, L_UNIFORM_PX = 5.0, 50.0
        radius_keys = ('radius', 'radius_px', 'radius_px_true')
        length_keys = ('length', 'length_true', 'path_length_px')
        touched_keys = []
        if override_R:
            touched_keys.extend(radius_keys)
        if override_L:
            touched_keys.extend(length_keys)
        saved_geom: Dict[Tuple[int, int], dict] = {}
        if touched_keys:
            for u, v, d in self.G.edges(data=True):
                saved_geom[(u, v)] = {k: d.get(k) for k in touched_keys}
                if override_R:
                    for k in radius_keys:
                        d[k] = R_UNIFORM_PX
                if override_L:
                    for k in length_keys:
                        d[k] = L_UNIFORM_PX

        # A-stem conductance equalization.  For each A node, walk to
        # the first branch and set every edge in that stem to a
        # per-stem-uniform radius such that Σ 8μL/(πR⁴) = target_R,
        # where target_R is the median across all A stems.  Lengths
        # preserved.  Applied AFTER any uniform-R/L override so the
        # stems get the correct per-stem radius rather than the
        # uniform value.
        equalize_A = bool(self._sim_equalize_A_chk.isChecked())
        if equalize_A and self._source_nodes:
            from ..analyze.constants import PX_SIZE_UM
            from ..analyze.constants import MU_DEFAULT
            px_to_m = PX_SIZE_UM * 1e-6
            stems = {a: self._find_inflow_stem(a)
                     for a in self._source_nodes}
            resistances = {a: self._stem_poiseuille_resistance(e)
                           for a, e in stems.items()}
            finite_R = [r for r in resistances.values()
                        if r > 0 and np.isfinite(r)]
            if finite_R:
                target_R = float(np.median(finite_R))
                for a, edges in stems.items():
                    if not edges:
                        continue
                    L_total_m = 0.0
                    for (u, v) in edges:
                        L_px = self.G.edges[u, v].get('length')
                        if L_px is None:
                            continue
                        try:
                            L_total_m += float(L_px) * px_to_m
                        except (TypeError, ValueError):
                            continue
                    if L_total_m <= 0 or target_R <= 0:
                        continue
                    R_new_m = (8.0 * MU_DEFAULT * L_total_m
                                / (np.pi * target_R)) ** 0.25
                    R_new_px = R_new_m / px_to_m
                    for (u, v) in edges:
                        key = (u, v) if self.G.has_edge(u, v) else (v, u)
                        if key not in saved_geom:
                            saved_geom[key] = {}
                        ed = self.G.edges[u, v]
                        for rk in radius_keys:
                            if rk not in saved_geom[key]:
                                saved_geom[key][rk] = ed.get(rk)
                            ed[rk] = R_new_px

        # V-side BC mode (combo index → behaviour):
        #   0 → Q at V (measured)            — standard Q-BC everywhere
        #   1 → P-match V at DC + AC rescaled — Option B.  Pass 1 sets
        #        P=0 at every V to predict Q̄_V from physics.  Pass 2
        #        prescribes per-V AC = (|measured_n|/|measured_dc|)
        #        × Q̄_V_predicted, with measured phases — i.e. the
        #        mean-flow-normalised measured amplitudes used as
        #        constraints, scaled by the model's DC prediction.
        v_bc_mode = 0   # V-side P-match removed with Norton closure

        def _q_into_node(edge_flows_dict, node):
            """Sum Q on incident edges, signed so result = Q flowing
            INTO `node`.  edge_flows is keyed (u, v) with +Q along
            u→v."""
            n_harmonics = 3
            acc = np.zeros(n_harmonics + 1, dtype=complex)
            for nbr in self.G.neighbors(node):
                if (nbr, node) in edge_flows_dict:
                    acc += np.asarray(edge_flows_dict[(nbr, node)],
                                       dtype=complex)
                elif (node, nbr) in edge_flows_dict:
                    acc -= np.asarray(edge_flows_dict[(node, nbr)],
                                       dtype=complex)
            return acc

        try:
            # Lazy import — keeps viewer startup cheap and avoids
            # pulling in the heavy adaptation module circular chain.
            from plexusmech.tl.transmission_line import solve_transmission_line  # ported 2026-09-30
            t0 = time.time()
            if equal_flux_A:
                # Two-pass equal-flux:
                #   Pass 1: Q(A_i) = Q_A_total / N_A prescribed, P(V) = 0
                #           at every harmonic.  Extract Q̂_DC(V) at each
                #           V node via edge_flows sum.  AC at V from this
                #           pass is a model output (network response to
                #           the AC at A alone).
                #   Pass 2: Q(A_i) unchanged; Q(V_j) = Q̂_DC(V_j) ·
                #           measured PI ratio at V_j (all harmonics).
                #           V AC waveform is now the trusted measurement,
                #           scaled to the DC the geometry predicts.
                Q_A_total = float(getattr(self, '_sim_qa_spin', None).value()
                                    if getattr(self, '_sim_qa_spin', None)
                                    else 0.0)
                N_A = max(1, len(self._source_nodes))
                Q_A_per_port = Q_A_total / N_A

                # A-side BC (same in both passes): PI-preserving from measurement
                def _build_A_bc():
                    out: Dict[int, np.ndarray] = {}
                    for n in self._source_nodes:
                        target = np.zeros(4, dtype=complex)
                        target[0] = complex(Q_A_per_port)
                        bch = self.G.nodes[n].get('bc_harmonics')
                        if bch is not None:
                            bch = np.asarray(bch, dtype=complex)
                            DC_meas = bch[0].real
                            if abs(DC_meas) > 1e-30:
                                for k in range(1, 4):
                                    if k < len(bch):
                                        target[k] = complex(Q_A_per_port) * (bch[k] / DC_meas)
                        out[n] = target
                    return out

                # --- Pass 1: Q at A only, V grounded ---
                # Pass 1 always grounds V at P=0 to give a clean geometry
                # prediction of Q_DC at V.  The V-closure choice only
                # affects Pass 2's V handling.
                bc_override_p1 = _build_A_bc()
                result_p1 = solve_transmission_line(
                    self.G, D=D, n_harmonics=3, f0_hz=f0,
                    bc_harmonics_override=bc_override_p1,
                    sink_pressure_bc=np.zeros(4, dtype=complex),
                    verbose=False,
                )
                # Same V-closure sub-modes as Dirichlet-tie:
                v_closure_pressure_eq = bool(
                    getattr(self, '_sim_v_closure_pressure', None)
                    is not None
                    and self._sim_v_closure_pressure.isChecked())
                v_closure_tied_eq = bool(
                    getattr(self, '_sim_v_closure_tied', None)
                    is not None
                    and self._sim_v_closure_tied.isChecked())

                # DC-BC diagnostic — Pass-1 predicted V DC vs measured
                if bool(getattr(self, '_sim_dc_diag_chk', None) is not None
                        and self._sim_dc_diag_chk.isChecked()):
                    _print_dc_bc_diagnostic(
                        self.G, self._source_nodes, self._sink_nodes,
                        result_p1)

                # --- Build Pass 2 Q-BC ---
                # 'flux': add Q-BC at V (measured PI × predicted DC).
                # 'pressure' / 'tied': keep Q-BC only at A; V handling is
                #                       done via sink_pressure_bc or shim.
                # 'meas_ac': Q-BC at V with target[k] = measured bch[k].
                v_closure_meas_ac_eq = bool(
                    getattr(self, '_sim_v_closure_meas_ac', None) is not None
                    and self._sim_v_closure_meas_ac.isChecked())
                bc_override_p2: Dict[int, np.ndarray] = dict(bc_override_p1)  # A unchanged
                if not (v_closure_pressure_eq or v_closure_tied_eq):
                    for n in self._sink_nodes:
                        bch = self.G.nodes[n].get('bc_harmonics')
                        if bch is None:
                            continue
                        bch = np.asarray(bch, dtype=complex)
                        q_pred = _actual_boundary_Q_dc(
                            self.G, n, result_p1.edge_flows)
                        if q_pred is None:
                            continue
                        target = np.zeros(4, dtype=complex)
                        target[0] = complex(q_pred)
                        if v_closure_meas_ac_eq:
                            # V AC = measured phasor DIRECTLY.
                            for k in range(1, 4):
                                if k < len(bch):
                                    target[k] = bch[k]
                        else:
                            DC_meas = bch[0].real
                            if abs(DC_meas) > 1e-30:
                                for k in range(1, 4):
                                    if k < len(bch):
                                        target[k] = complex(q_pred) * (bch[k] / DC_meas)
                        bc_override_p2[n] = target

                # --- Pass 2 solve ---
                if v_closure_tied_eq and len(self._sink_nodes) >= 2:
                    result = solve_transmission_line(
                        self.G, D=D, n_harmonics=3, f0_hz=f0,
                        bc_harmonics_override=bc_override_p2,
                        tied_pressure_nodes=list(self._sink_nodes),
                        verbose=False,
                    )
                elif v_closure_pressure_eq:
                    result = solve_transmission_line(
                        self.G, D=D, n_harmonics=3, f0_hz=f0,
                        bc_harmonics_override=bc_override_p2,
                        sink_pressure_bc=np.zeros(4, dtype=complex),
                        verbose=False,
                    )
                else:
                    result = solve_transmission_line(
                        self.G, D=D, n_harmonics=3, f0_hz=f0,
                        bc_harmonics_override=bc_override_p2,
                        verbose=False,
                    )

                # A-pressure diagnostic: what pressures did geometry need
                # at each A to deliver equal flux?  Read from Pass 2's
                # node_pressures (Pass 1 has the same A pressures).
                if bool(getattr(self, '_sim_dc_diag_chk', None) is not None
                        and self._sim_dc_diag_chk.isChecked()):
                    print()
                    print('=' * 70)
                    print(f'  Equal-flux BC: Q(A_i) = {Q_A_per_port:+.4f} '
                          f'nL/s at each of {N_A} A ports  '
                          f'(Q_A_total = {Q_A_total:+.4f})')
                    print('  A pressures (solver output — should be near-')
                    print('  equal if geometry is symmetric between A paths):')
                    node_P = getattr(result_p1, 'node_pressures', None)
                    for n in self._source_nodes:
                        P_dc = None
                        if isinstance(node_P, dict):
                            pv = node_P.get(n)
                            if pv is not None:
                                P_dc = float(np.asarray(pv, dtype=complex)[0].real)
                        if P_dc is None:
                            print(f'    node {n:>6}:  (pressure not in result)')
                        else:
                            print(f'    node {n:>6}:  P_DC = {P_dc:+.4f} Pa')
                    print('=' * 70)
                    print()

            elif dirichlet_tie:
                # New default: V nodes Dirichlet-tied to common P_SV.
                # Two-pass approach:
                #   Pass 1: pure DC Dirichlet at ΔP (no AC pressure BC).
                #           Extract Q_0 at every boundary from the DC solve.
                #   Pass 2: fully Neumann Q at every boundary.  Each Q at
                #           harmonic k is the per-node measured normalized
                #           spectrum (H_k / H_0) rescaled by that node's
                #           Q_0 from pass 1.  This guarantees Q at every
                #           boundary has the same H_k/H_0 ratio as measured
                #           while DC is driven by ΔP.
                dP = float(getattr(self, '_sim_dp_spin', None).value()
                            if getattr(self, '_sim_dp_spin', None) else 0.0)

                if dP > 0.0:
                    # --- Pass 1: DC Dirichlet, no AC drive ---
                    p_sink_p1 = np.zeros(4, dtype=complex)
                    p_source_p1 = np.zeros(4, dtype=complex)
                    p_source_p1[0] = complex(dP)
                    result_p1 = solve_transmission_line(
                        self.G, D=D, n_harmonics=3, f0_hz=f0,
                        sink_pressure_bc=p_sink_p1,
                        source_pressure_bc=p_source_p1,
                        verbose=False,
                    )

                    # V-closure sub-mode: 'flux' pins V (current behaviour),
                    # 'pressure' leaves V at Dirichlet P=0 so V flows stay
                    # a network prediction rather than a BC.
                    # 'tied' ties P_V1(ω) = P_V2(ω) at every harmonic via
                    # a phantom shim edge; V AC pressure becomes a model
                    # output with the shared-atrium constraint.
                    v_closure_pressure = bool(
                        getattr(self, '_sim_v_closure_pressure', None)
                        is not None
                        and self._sim_v_closure_pressure.isChecked())
                    v_closure_tied = bool(
                        getattr(self, '_sim_v_closure_tied', None)
                        is not None
                        and self._sim_v_closure_tied.isChecked())
                    v_closure_meas_ac = bool(
                        getattr(self, '_sim_v_closure_meas_ac', None)
                        is not None
                        and self._sim_v_closure_meas_ac.isChecked())

                    # --- Diagnostic: Pass-1 predicted vs measured DC per port ---
                    if bool(getattr(self, '_sim_dc_diag_chk', None) is not None
                            and self._sim_dc_diag_chk.isChecked()):
                        _print_dc_bc_diagnostic(
                            self.G, self._source_nodes, self._sink_nodes,
                            result_p1)

                    # --- Build boundary Q for Pass 2 ---
                    # Boundary invariant we trust: the MEASURED PI ratio
                    # bch[k]/bch[0] at each port.  To make sure the
                    # INJECTED PI (target[k]/target[0]) equals the measured
                    # PI, we scale the AC target by target[0] itself:
                    #   target[0] = Q̂_DC(n)                          (predicted DC)
                    #   target[k] = target[0] · bch[k]/bch[0]         (k ≥ 1)
                    # Then injected PI = target[k]/target[0] = bch[k]/bch[0]
                    # = measured PI, by construction.  If we injected bch[k]
                    # DIRECTLY (bypassing the scale), the injected PI would
                    # equal bch[k]/target[0] = measured_AC/predicted_DC,
                    # which is NOT the measured PI when Q̂_DC ≠ Re(bch[0]).
                    # 'pressure': Q-BC at A only, V held Dirichlet P=0.
                    # 'tied':     Q-BC at A only, V AC free but tied via penalty.
                    # 'flux':     Q-BC at both A and V (V AC = Q̂_DC · bch[k]/bch[0]).
                    # 'meas_ac':  Q-BC at both A and V, V AC target = bch[k]
                    #             DIRECTLY (measured phasor, no PI rescaling).
                    v_only_A_qbc = v_closure_pressure or v_closure_tied
                    which_nodes = list(self._source_nodes)
                    if not v_only_A_qbc:
                        which_nodes += list(self._sink_nodes)
                    bc_override_p2: Dict[int, np.ndarray] = {}
                    for n in which_nodes:
                        bch = self.G.nodes[n].get('bc_harmonics')
                        if bch is None:
                            continue
                        bch = np.asarray(bch, dtype=complex)
                        # Use edge_flows sum for actual pass-1 predicted DC
                        # (not the echoed boundary_Q_harmonics field).
                        q_pred = _actual_boundary_Q_dc(
                            self.G, n, result_p1.edge_flows)
                        if q_pred is None:
                            continue
                        target = np.zeros(4, dtype=complex)
                        target[0] = complex(q_pred)           # predicted DC
                        is_v = n in self._sink_nodes
                        if is_v and v_closure_meas_ac:
                            # V AC = measured phasor DIRECTLY (untrusted DC
                            # replaced with prediction, trusted AC kept as-is).
                            for k in range(1, 4):
                                if k < len(bch):
                                    target[k] = bch[k]
                        else:
                            # Default (A nodes; V under 'flux' closure): AC =
                            # predicted DC × measured PI ratio.
                            DC_meas = bch[0].real
                            if abs(DC_meas) > 1e-30:
                                for k in range(1, 4):
                                    if k < len(bch):
                                        target[k] = complex(q_pred) * (bch[k] / DC_meas)
                        bc_override_p2[n] = target

                    # --- Pass 2 solve ---
                    if v_closure_tied and len(self._sink_nodes) >= 2:
                        # Tied V: penalty admittance in solver ties V
                        # pressures at every harmonic; V nodes get no
                        # Q-BC (V flows are model outputs).
                        result = solve_transmission_line(
                            self.G, D=D, n_harmonics=3, f0_hz=f0,
                            bc_harmonics_override=bc_override_p2,
                            tied_pressure_nodes=list(self._sink_nodes),
                            verbose=False,
                        )
                    elif v_closure_pressure:
                        # Grounded venous: keep P_V = 0 at all harmonics
                        result = solve_transmission_line(
                            self.G, D=D, n_harmonics=3, f0_hz=f0,
                            bc_harmonics_override=bc_override_p2,
                            sink_pressure_bc=np.zeros(4, dtype=complex),
                            verbose=False,
                        )
                    else:
                        # Fully Neumann Q at every boundary (default)
                        result = solve_transmission_line(
                            self.G, D=D, n_harmonics=3, f0_hz=f0,
                            bc_harmonics_override=bc_override_p2,
                            verbose=False,
                        )
                else:
                    # Flux-driven fallback: Q at A from measured/rescaled
                    # bc_override, V held at Dirichlet common P=0 (AC=0
                    # too — pure zero pressure at V; no AC pressure drive).
                    a_only = {n: arr for n, arr in bc_override.items()
                              if n in self._source_nodes}
                    result = solve_transmission_line(
                        self.G, D=D, n_harmonics=3, f0_hz=f0,
                        bc_harmonics_override=a_only,
                        sink_pressure_bc=np.zeros(4, dtype=complex),
                        verbose=False,
                    )
            elif v_bc_mode == 0:
                # Standard: prescribe Q at every boundary.
                result = solve_transmission_line(
                    self.G, D=D, n_harmonics=3, f0_hz=f0,
                    bc_harmonics_override=bc_override,
                    verbose=False,
                )
            else:
                # Option B: P-match V at DC, rescale per-V AC by the
                # predicted Q̄_V × measured per-V PI/phase.
                # Pass 1: predict Q̄_V per V.
                a_only_override = {
                    n: arr for n, arr in bc_override.items()
                    if n in self._source_nodes
                }
                result_pass1 = solve_transmission_line(
                    self.G, D=D, n_harmonics=3, f0_hz=f0,
                    bc_harmonics_override=a_only_override,
                    sink_pressure_bc=0.0,
                    verbose=False,
                )
                # Build pass-2 override: A's unchanged, V's get DC =
                # predicted (signed into-network) and AC = ratio ×
                # predicted DC, with the ratio taken from each V's
                # MEASURED bc_harmonics so the per-V PI and phase
                # are the data-side constraint.
                pass2_override: Dict[int, np.ndarray] = dict(a_only_override)
                for sn in self._sink_nodes:
                    measured = self.G.nodes[sn].get('bc_harmonics')
                    if measured is None:
                        continue
                    measured = np.asarray(measured, dtype=complex)
                    q_into_v = _q_into_node(result_pass1.edge_flows, sn)
                    # Solver's "Q at V" convention is +inflow into the
                    # network at V = −outflow_from_network_at_V.
                    # `q_into_v` is the predicted Q flowing INTO V
                    # from the network (= outflow_from_network), so
                    # the solver-convention Q_V is −q_into_v.
                    Q_v_dc_new = -q_into_v[0].real
                    new_arr = np.zeros_like(measured)
                    new_arr[0] = Q_v_dc_new
                    # Ratio Q_n / Q_dc is convention-independent (any
                    # uniform sign flip cancels).  Multiplying by the
                    # new DC rescales the AC magnitude while keeping
                    # the measured per-V phase intact.
                    if abs(measured[0]) > 1e-12 and abs(Q_v_dc_new) > 1e-30:
                        for k in range(1, len(measured)):
                            new_arr[k] = (measured[k] / measured[0]
                                           ) * Q_v_dc_new
                    pass2_override[sn] = new_arr
                result = solve_transmission_line(
                    self.G, D=D, n_harmonics=3, f0_hz=f0,
                    bc_harmonics_override=pass2_override,
                    verbose=False,
                )
            elapsed = time.time() - t0
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._sim_status_lbl.setText(
                f"<small><font color='#c33'>Solver failed: "
                f"{type(e).__name__}: {e}</font></small>")
            self._sim_run_btn.setEnabled(True)
            return
        finally:
            # Always restore the graph's original geometry so the
            # mutation is invisible to the rest of the viewer.
            if saved_geom:
                for (u, v), saved in saved_geom.items():
                    for k, original in saved.items():
                        if original is None:
                            self.G.edges[u, v].pop(k, None)
                        else:
                            self.G.edges[u, v][k] = original

        n = self._store_sim_result(result, f0, D, override_R,
                                   override_L, R_UNIFORM_PX, L_UNIFORM_PX)
        # Stamp the graph with the run parameters so the state can be
        # saved + reloaded with the UI restored.  Any subsequent 'Save
        # graph state…' click will pickle this alongside the sim data.
        self._stamp_sim_meta(D=D, f0=f0, n_edges=n, elapsed_s=elapsed)
        self._sim_status_lbl.setText(
            f"<small>OK · {elapsed:.1f}s · {n} edges<br>"
            f"D = {D:.3g}, f₀ = {f0:.3f} Hz<br>"
            f"Switch the View-tab Source to <b>Simulated</b> to view "
            f"sim fields; P quantity becomes available.</small>")
        self._sim_run_btn.setEnabled(True)
        self._sim_clear_btn.setEnabled(True)


    # ── Sim persistence: stamp / save / restore ─────────────────────
    def _stamp_sim_meta(self, D, f0, n_edges, elapsed_s):
        """Write G.graph['sim_meta'] snapshotting the UI parameters
        used for this run.  Read back by _restore_sim_ui_from_graph()
        on next viewer launch to repopulate the controls."""
        from datetime import datetime, timezone
        dp = float(self._sim_dp_spin.value()
                    if getattr(self, '_sim_dp_spin', None) else 0.0)
        qat = float(self._sim_qa_spin.value()
                     if getattr(self, '_sim_qa_spin', None) else 1.0)
        bc_mode = ('dirichlet_tie'
                    if (getattr(self, '_sim_bc_radio_dirichlet', None)
                        and self._sim_bc_radio_dirichlet.isChecked())
                    else 'equalflux')
        if (getattr(self, '_sim_v_closure_meas_ac', None)
                and self._sim_v_closure_meas_ac.isChecked()):
            v_closure = 'meas_ac'
        elif (getattr(self, '_sim_v_closure_pressure', None)
                and self._sim_v_closure_pressure.isChecked()):
            v_closure = 'pressure'
        elif (getattr(self, '_sim_v_closure_tied', None)
                and self._sim_v_closure_tied.isChecked()):
            v_closure = 'tied'
        else:
            v_closure = 'flux'
        self.G.graph['sim_meta'] = dict(
            timestamp=datetime.now(timezone.utc).isoformat(),
            D_areal=float(D),
            eta=float(getattr(self, '_sim_last_eta', 0.0)),
            wall_model='constant_loss_tangent',
            f0_hz=float(f0),
            K_harmonics=3,
            bc_mode=bc_mode,
            v_closure=v_closure,
            dp_target_Pa=dp,
            q_a_total_nL_s=qat,
            n_edges_written=int(n_edges),
            elapsed_s=float(elapsed_s),
        )


    def _on_save_state_clicked(self):
        """Pickle the whole in-memory graph to <stem>_state.gpickle.
        Bundles normalization fields, sim results, tile_gauges, and
        the sim_meta snapshot in one file."""
        from qtpy.QtWidgets import QFileDialog, QMessageBox
        default_name = self.graph_path.with_name(
            self.graph_path.stem + '_state.gpickle')
        path, _ = QFileDialog.getSaveFileName(
            self.viewer.window._qt_window,
            "Save graph state (normalized + sim + metadata)",
            str(default_name),
            "Pickled graph (*.gpickle);;All files (*)")
        if not path:
            return
        try:
            import pickle
            with open(path, 'wb') as f:
                self._dump_graph(f)
            has_sim = bool(self.G.graph.get('sim_meta'))
            has_norm = bool(self.G.graph.get('tile_gauges'))
            QMessageBox.information(
                self.viewer.window._qt_window, "Saved",
                f"Wrote {path}\n"
                f"  normalized fields: {'yes' if has_norm else 'no'}\n"
                f"  sim fields:        {'yes' if has_sim else 'no'}")
        except Exception as exc:
            QMessageBox.critical(
                self.viewer.window._qt_window, "Save failed",
                f"Could not save graph:\n{exc}")


    def _restore_sim_ui_from_graph(self):
        """If G.graph carries a sim_meta blob (i.e. the graph was
        saved after a previous simulation run), set the Simulate tab
        controls to match and populate _sim_loaded_lbl so the user can
        see what was run."""
        meta = self.G.graph.get('sim_meta')
        lbl = getattr(self, '_sim_loaded_lbl', None)
        if not meta or lbl is None:
            return
        # Radios
        bc_mode = meta.get('bc_mode', 'dirichlet_tie')
        if bc_mode == 'equalflux' and getattr(self, '_sim_bc_radio_equalflux', None):
            self._sim_bc_radio_equalflux.setChecked(True)
        elif getattr(self, '_sim_bc_radio_dirichlet', None):
            self._sim_bc_radio_dirichlet.setChecked(True)
        vc = meta.get('v_closure', 'meas_ac')
        vc_radio = {
            'flux':     getattr(self, '_sim_v_closure_flux', None),
            'pressure': getattr(self, '_sim_v_closure_pressure', None),
            'tied':     getattr(self, '_sim_v_closure_tied', None),
            'meas_ac':  getattr(self, '_sim_v_closure_meas_ac', None),
        }.get(vc)
        if vc_radio is not None:
            vc_radio.setChecked(True)
        # Spinboxes
        if getattr(self, '_sim_dp_spin', None) and 'dp_target_Pa' in meta:
            self._sim_dp_spin.setValue(float(meta['dp_target_Pa']))
        if getattr(self, '_sim_qa_spin', None) and 'q_a_total_nL_s' in meta:
            self._sim_qa_spin.setValue(float(meta['q_a_total_nL_s']))
        # Label
        ts = str(meta.get('timestamp', '?'))[:19].replace('T', ' ')
        # ΔP is only used in dirichlet_tie mode.  In equalflux (the only
        # remaining active mode) it's ignored by the solve, so omit it
        # from the label to avoid the misleading "10 Pa" that came from
        # the hidden default spinbox value.
        _line2 = (f"D = {meta.get('D_areal', float('nan')):.3g},  "
                  f"f₀ = {meta.get('f0_hz', float('nan')):.3f} Hz,  "
                  f"Q_A_total = {meta.get('q_a_total_nL_s', float('nan')):.3f} nL/s")
        if meta.get('bc_mode') == 'dirichlet_tie':
            _line2 += f",  ΔP = {meta.get('dp_target_Pa', float('nan')):.2f} Pa"
        lbl.setText(
            f"<b>Loaded saved simulation</b>  ({ts})<br>"
            f"{_line2}<br>"
            f"BC = <tt>{meta.get('bc_mode','?')}</tt>,  "
            f"V closure = <tt>{meta.get('v_closure','?')}</tt><br>"
            f"<i>Sim fields already populated on the graph.  Switch View "
            f"tab Source to Simulated to view.  Click <b>Run</b> to "
            f"re-execute with any parameter change.</i>")
        lbl.show()
        # Turn on the Clear button since sim data is present
        if getattr(self, '_sim_clear_btn', None) is not None:
            self._sim_clear_btn.setEnabled(True)


    def _apply_pi_flip_and_persist_sim(self, mode='intact'):
        """Per-tile π-flip resolution on sim H1/H2/H3 (each independent),
        matching what sync_v2 did to measured. Then write persistent
        sim fields (`_h_Z_H{k}_sim`, `_h_phase_H{k}_sim`, `mean_Q_sim`,
        `amp_Q_sim`, `phase_sim`) so displays / overlays / save-state
        all see the flipped values.

        mode='intact': reads `_sim_tmp_harmonics` (complex [DC,H1,H2,H3])
        mode='carved': reads `_h_amp_H{k}_sim_carved` + `_h_phase_H{k}_sim_carved`
                       + `_h_amp_DC_sim_carved`
        Reference in both modes: `_h_amp_H{k}_norm` * exp(i·`_h_phase_H{k}_norm`).
        """
        # 1) Gather per-edge (sim, meas) H1/H2/H3 phasors + tile ids.
        def _tile_of(d):
            tms = d.get('measurements_piv') or []
            if isinstance(tms, list) and tms and isinstance(tms[0], dict):
                for f in ('tile_id', 'tid', 'vid'):
                    if f in tms[0]: return tms[0][f]
            return None
        def _meas_Z(d, k):
            a = d.get(f'_h_amp_H{k}_norm')
            p = d.get(f'_h_phase_H{k}_norm')
            if a is None or p is None or not np.isfinite(a) or not np.isfinite(p):
                return None
            return complex(float(a) * np.exp(1j * float(p)))
        def _sim_Z(d, k):
            if mode == 'intact':
                arr = d.get('_sim_tmp_harmonics')
                if arr is None: return None
                arr = np.asarray(arr, dtype=complex)
                return complex(arr[k]) if k < len(arr) else None
            # carved
            a = d.get(f'_h_amp_H{k}_sim_carved')
            p = d.get(f'_h_phase_H{k}_sim_carved')
            if a is None or p is None or not np.isfinite(a) or not np.isfinite(p):
                return None
            return complex(float(a) * np.exp(1j * float(p)))
        # SNR weight (edge-level)
        def _snr_w(d):
            for tm in (d.get('measurements_piv') or []):
                if isinstance(tm, dict):
                    s = tm.get('SNR_H1_dB')
                    if s is not None and np.isfinite(s):
                        return 10 ** (float(s) / 10.0)
            return 1.0
        # 2) Build per-tile flip maps for k=1,2,3 (each independent)
        flip = {1: {}, 2: {}, 3: {}}
        by_tile = {}
        for u, v, d in self.G.edges(data=True):
            t = _tile_of(d)
            if t is None: continue
            by_tile.setdefault(t, []).append((u, v, d))
        for t, edges in by_tile.items():
            if len(edges) < 2: continue
            for k in (1, 2, 3):
                r0 = 0.0; r1 = 0.0; n_ok = 0
                for u, v, d in edges:
                    zs = _sim_Z(d, k); zm = _meas_Z(d, k)
                    if zs is None or zm is None: continue
                    w = _snr_w(d)
                    r0 += w * abs(zs - zm)**2
                    r1 += w * abs(-zs - zm)**2
                    n_ok += 1
                if n_ok < 2: continue
                flip[k][t] = 0.0 if r0 <= r1 else np.pi
        # 3) Apply flips and write persistent sim fields
        n_flipped_H1 = sum(1 for f in flip[1].values() if f > 0)
        for u, v, d in self.G.edges(data=True):
            t = _tile_of(d)
            # DC magnitude (no flip)
            if mode == 'intact':
                arr = d.get('_sim_tmp_harmonics')
                if arr is None: continue
                arr = np.asarray(arr, dtype=complex).copy()
                # apply per-harmonic flips
                if len(arr) > 1 and t in flip[1]:
                    arr[1] = arr[1] * np.exp(1j * flip[1][t])
                if len(arr) > 2 and t in flip[2]:
                    arr[2] = arr[2] * np.exp(1j * flip[2][t])
                if len(arr) > 3 and t in flip[3]:
                    arr[3] = arr[3] * np.exp(1j * flip[3][t])
                d['_sim_tmp_harmonics'] = arr
                # Persistent write (matches lock_D6p5e4_eta0p4.py convention)
                d['_h_Z_DC_sim'] = complex(arr[0])
                d['mean_Q_sim'] = float(np.real(arr[0]))
                for k in range(1, min(4, len(arr))):
                    d[f'_h_Z_H{k}_sim'] = complex(arr[k])
                    d[f'_h_phase_H{k}_sim'] = float(np.angle(arr[k]))
                    if k == 1:
                        d['amp_Q_sim'] = float(abs(arr[1]))
                        d['phase_sim'] = float(np.angle(arr[1]))
                        # Update _sim_tmp_phase_Q to reflect the flip
                        d['_sim_tmp_phase_Q'] = float(np.angle(arr[1]))
            else:  # carved
                # Apply flip to the carved amp/phase pair (amp unchanged; phase += flip)
                for k in (1, 2, 3):
                    if t not in flip[k]: continue
                    ph_key = f'_h_phase_H{k}_sim_carved'
                    if ph_key in d and np.isfinite(d[ph_key]):
                        d[ph_key] = float(((d[ph_key] + flip[k][t] + np.pi) % (2*np.pi)) - np.pi)
                    zk = d.get(f'_h_Z_H{k}_sim_carved')
                    if zk is not None:
                        d[f'_h_Z_H{k}_sim_carved'] = complex(zk) * np.exp(1j * flip[k][t])
        n_tiles_total = len({t for t in by_tile.keys()})
        self.G.graph[f'pi_flip_stats_{mode}'] = dict(
            n_flipped_H1=int(n_flipped_H1),
            n_tiles=int(n_tiles_total),
        )
        try:
            print(f"[π-flip {mode}] H1: {n_flipped_H1}/{n_tiles_total} tiles flipped")
        except Exception:
            pass


    def _store_sim_result(self, result, f0, D, override_R=False,
                          override_L=False, R_UNIFORM_PX=5.0,
                          L_UNIFORM_PX=50.0):
        """Write a solve_transmission_line `result` into the per-edge
        `_sim_tmp_*` fields (mean_Q, amp_Q, PI, phase, pressure, harmonics…),
        activate the Simulated source, and refresh the View-tab combos.
        Shared by the Simulate tab and the Pump tab.  Returns #edges written."""
        # New sim data invalidates any cached A/V waveform references.
        self._av_ref_cache = None
        # Write per-edge sim results to the in-memory graph.  Use the
        # `_sim_tmp_` namespace so we don't collide with editing-app
        # outputs (`mean_Q_sim`, `PI_sim`, …) already stored on the
        # graph.
        #
        # Q fields are stored as magnitudes:
        #   mean_Q  →  |mean_Q|       (matches measured: mean_Q ≥ 0
        #                              by convention — see methods.md)
        #   amp_Q   →  |H₁(Q)|        (already a magnitude in the result)
        #
        # Two phases are exposed so the user can see both flow and
        # pressure timing per harmonic.  `result.phase` is the Q-phase
        # with the sign canonicalisation already applied by the solver
        # (flips when flow direction would otherwise be negative);
        # `result.pressure_phase` is the midpoint-pressure H₁ phase.
        n = 0
        for (u, v) in result.edge_flows.keys():
            if not self.G.has_edge(u, v):
                continue
            d = self.G.edges[u, v]
            mq = result.mean_Q.get((u, v), float('nan'))
            aq = result.amp_Q.get((u, v), float('nan'))
            d['_sim_tmp_mean_Q'] = (float(abs(mq)) if np.isfinite(mq)
                                    else float('nan'))
            d['_sim_tmp_amp_Q'] = (float(abs(aq)) if np.isfinite(aq)
                                    else float('nan'))
            d['_sim_tmp_PI'] = float(
                result.PI.get((u, v), float('nan')))
            d['_sim_tmp_phase_Q'] = float(
                result.phase.get((u, v), float('nan')))
            d['_sim_tmp_phase_P'] = float(
                result.pressure_phase.get((u, v), float('nan')))
            d['_sim_tmp_dissipation'] = float(
                result.dissipation.get((u, v), float('nan')))
            # Geometry overrides — written only when the corresponding
            # override mode was active, so the sim-mode View tab can
            # show the constant value.  Heterogeneous sim → keys
            # absent → fall through to the measured top-level attrs.
            if override_R:
                d['_sim_tmp_radius'] = R_UNIFORM_PX
            if override_L:
                d['_sim_tmp_length'] = L_UNIFORM_PX
            # Persist the full complex harmonic array (DC + H₁ + H₂ + H₃)
            # so `_plot_comparison` can reconstruct a synthetic Q(t)
            # when the user is viewing a sim field.  Sign-canonicalise
            # so mean_Q ≥ 0 (matches measured `mean_Q` convention from
            # methods.md), shifting all harmonic phases by π if needed.
            harm = result.edge_flows.get((u, v))
            if harm is not None and len(harm) >= 1:
                arr = np.asarray(harm, dtype=complex).copy()
                if np.isfinite(arr[0].real) and arr[0].real < 0:
                    arr = -arr
                d['_sim_tmp_harmonics'] = arr
                d['_sim_tmp_f0_hz'] = float(f0)
            # Midpoint pressure harmonics + summary scalars
            # (sim-only fields, available in the View tab when the
            # Simulated source is active).  Pressure PI is undefined
            # near the gauge node where |P_DC| → 0; we record nan
            # there rather than picking an arbitrary reference.
            p_u = result.node_pressures.get(u)
            p_v = result.node_pressures.get(v)
            if p_u is not None and p_v is not None:
                p_u_arr = np.asarray(p_u, dtype=complex)
                p_v_arr = np.asarray(p_v, dtype=complex)
                p_mid = 0.5 * (p_u_arr + p_v_arr)
                d['_sim_tmp_p_harmonics'] = p_mid
                p_dc = float(p_mid[0].real) if len(p_mid) > 0 else float('nan')
                amp_p = (float(abs(p_mid[1]))
                         if len(p_mid) > 1 else float('nan'))
                # Signed mean midpoint pressure (Pa).  Kept signed so
                # the gauge polarity is visible in the colormap; users
                # who only want magnitude can toggle log scale off and
                # read the sign directly.
                d['_sim_tmp_pressure_mean'] = p_dc
                d['_sim_tmp_pressure_amp'] = amp_p
                # H2 midpoint pressure amplitude (Pa) — analogous to
                # pressure_amp but at the second harmonic.  NaN if the
                # solver didn't compute H2.
                amp_p_h2 = (float(abs(p_mid[2]))
                             if len(p_mid) > 2 else float('nan'))
                d['_sim_tmp_pressure_amp_H2'] = amp_p_h2
                # Pressure drop across the edge: |P_u_DC − P_v_DC| (Pa).
                # Magnitude so the colormap reads unambiguously; sign
                # would depend on edge orientation (undirected graph).
                if len(p_u_arr) > 0 and len(p_v_arr) > 0:
                    d['_sim_tmp_pressure_drop'] = float(
                        abs(p_u_arr[0].real - p_v_arr[0].real))
            n += 1
        # ── π-flip resolution + persistent field write ─────────────────
        # After the raw sim is stored in `_sim_tmp_*`, apply the per-tile
        # π-flip on H1/H2/H3 (independently) using measured `_h_*_norm`
        # phasors as reference. This puts sim into the same sync_v2
        # amplitude-sign frame the measured field lives in, so the View
        # tab's phase and |H1|/|DC| displays are directly comparable to
        # measured. Also write persistent `_h_Z_H*_sim`, `mean_Q_sim`,
        # `amp_Q_sim`, `phase_sim` fields (matching lock_D6p5e4_eta0p4.py's
        # convention) so downstream scripts / overlays read from a common
        # namespace and the sim state survives save-and-reload.
        self._apply_pi_flip_and_persist_sim(mode='intact')
        self._sim_active = True
        self._sim_last_D = D
        self._sim_last_f0_hz = f0
        # Enable the Simulated source radio in the View tab.
        if getattr(self, '_src_radio_sim', None) is not None:
            self._src_radio_sim.setEnabled(True)
            self._src_radio_sim.setToolTip("")
        # Re-sync the field combos so the P quantity and sim-only
        # properties become enabled (or stay enabled).
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()
        return n


    # ── Ligation experiment helpers ───────────────────────────────────
    def _clear_simulation(self):
        """Strip `_sim_tmp_*` keys from every edge, fall back to the
        measured source, and refresh the View-tab UI."""
        sim_keys = list(SIM_FIELD_MAP.values()) + SIM_INTERNAL_KEYS
        removed = 0
        for _, _, d in self.G.edges(data=True):
            for k in sim_keys:
                if k in d:
                    del d[k]
                    removed += 1
        self._sim_active = False
        self._av_ref_cache = None
        # Force the View tab back to the measured source.
        self.current_source = 'measured'
        if getattr(self, '_src_radio_measured', None) is not None:
            self._src_radio_measured.blockSignals(True)
            self._src_radio_sim.blockSignals(True)
            try:
                self._src_radio_measured.setChecked(True)
                self._src_radio_sim.setEnabled(False)
            finally:
                self._src_radio_measured.blockSignals(False)
                self._src_radio_sim.blockSignals(False)
        # Resync the field combos (P quantity + sim-only properties
        # disappear / get greyed out) and re-render.
        self._sync_field_combos()
        self._refresh_edges()
        self._refresh_nodes()
        self._refresh_cbar()
        self._sim_clear_btn.setEnabled(False)
        self._sim_status_lbl.setText(
            f"<small>Cleared {removed} sim values from "
            f"{self.G.number_of_edges()} edges.</small>")


    def _carved_sim_data_available(self) -> bool:
        """True if any edge carries `_h_*_sim_carved` fields from a
        previous Carved Sim run."""
        try:
            for _, _, d in self.G.edges(data=True):
                if any(k.endswith('_sim_carved') for k in d.keys()):
                    return True
            return False
        except Exception:
            return False


    def _refresh_carved_sim_radio_enable(self):
        """Called after a Carved Sim run to unlock the radio option."""
        try:
            if hasattr(self, '_src_radio_sim_carved'):
                avail = self._carved_sim_data_available()
                self._src_radio_sim_carved.setEnabled(avail)
                if avail:
                    self._src_radio_sim_carved.setToolTip("")
        except Exception:
            pass


    # ── Custom-BC helpers ────────────────────────────────────────────
    def _on_bc_source_change(self, _checked=None):
        """Toggle the Custom-waveform spinboxes based on BC source.
        The preview canvas stays live in both modes — it always shows
        whatever waveform would actually be applied at solve time."""
        if not hasattr(self, '_sim_custom_group'):
            return
        is_custom = bool(self._sim_bc_radio_custom.isChecked())
        # Show the custom-waveform grid only in Custom mode (keeps the tab
        # compact under the default class-waveform / measured modes).
        self._sim_custom_group.setVisible(is_custom)
        for sp in getattr(self, '_sim_custom_spins', {}).values():
            sp.setEnabled(is_custom)
        # Re-render the preview for the new mode.
        self._refresh_custom_preview()


    def _read_custom_waveform(self):
        """Return per-side (amps, phases) tuples for the A and V custom
        waveform specs.  Amplitudes are |H_k|/|Q̄| (length 3, k=1..3);
        phases are in radians (length 3).
        Returns: ((amps_A, phases_A), (amps_V, phases_V)).
        """
        out = {}
        for side in ('A', 'V'):
            amps = []
            phases = []
            for k in (1, 2, 3):
                a = float(
                    self._sim_custom_spins[(side, k, 'amp')].value())
                p = np.radians(float(
                    self._sim_custom_spins[(side, k, 'phase')].value()))
                amps.append(a)
                phases.append(p)
            out[side] = (amps, phases)
        return out['A'], out['V']


    def _refresh_custom_preview(self):
        """Refresh the BC waveform preview.

        Shows whichever waveform would actually be applied at solve
        time — custom (from the spinboxes) or measured (per-type
        complex average of stored bc_harmonics, with the canonical
        −1 sign-flip applied to sinks).  Y-axis is absolute Q(t) in
        nL/s so the A↔V magnitude asymmetry reads honestly.
        """
        if not hasattr(self, '_sim_custom_fig'):
            return
        f0 = (float(self._sim_detected_f0)
              if np.isfinite(self._sim_detected_f0) else 2.5)
        T_period = 1.0 / f0
        t = np.linspace(0.0, 2.0 * T_period, 400)
        omega = 2.0 * np.pi * f0
        target_flux = float(self._sim_flux_spin.value())
        is_custom = self._sim_bc_radio_custom.isChecked()
        n_src = max(1, len(self._source_nodes))
        n_snk = max(1, len(self._sink_nodes))

        self._sim_custom_fig.clear()
        ax = self._sim_custom_fig.add_subplot(111)

        if is_custom:
            (amps_A, phases_A), (amps_V, phases_V) = \
                self._read_custom_waveform()
            Qbar_A = +target_flux / n_src
            Qbar_V = -target_flux / n_snk
            y_A = np.full_like(t, Qbar_A)
            y_V = np.full_like(t, Qbar_V)
            for k_idx, (a, p) in enumerate(
                    zip(amps_A, phases_A), start=1):
                y_A += a * Qbar_A * np.cos(k_idx * omega * t + p)
            for k_idx, (a, p) in enumerate(
                    zip(amps_V, phases_V), start=1):
                y_V += a * Qbar_V * np.cos(k_idx * omega * t + p)
            PI_A = 2.0 * abs(amps_A[0]) if amps_A[0] else 0.0
            PI_V = 2.0 * abs(amps_V[0]) if amps_V[0] else 0.0
            title = (f"CUSTOM  Q̄_A = {abs(Qbar_A):.3f}  "
                     f"(PI_A ≈ {PI_A:.2f}),  "
                     f"Q̄_V = {abs(Qbar_V):.3f} nL/s  "
                     f"(PI_V ≈ {PI_V:.2f})")
        else:
            # Measured: per-type complex average of stored bc_harmonics,
            # with canonical sign-flip on sinks (Q into network).
            src_arrs = [np.asarray(self.G.nodes[n].get('bc_harmonics'),
                                     dtype=complex)
                        for n in self._source_nodes
                        if self.G.nodes[n].get('bc_harmonics') is not None]
            snk_arrs = [-np.asarray(self.G.nodes[n].get('bc_harmonics'),
                                      dtype=complex)
                        for n in self._sink_nodes
                        if self.G.nodes[n].get('bc_harmonics') is not None]
            if not src_arrs or not snk_arrs:
                ax.text(0.5, 0.5,
                        'No measured bc_harmonics on this graph.',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=10, color='#888')
                ax.set_xticks([]); ax.set_yticks([])
                self._sim_custom_canvas.draw_idle()
                return
            src_avg = np.mean(np.stack(src_arrs, axis=0), axis=0).copy()
            snk_avg = np.mean(np.stack(snk_arrs, axis=0), axis=0).copy()
            # Apply the same flux scaling the solver path uses.
            if self._sim_equal_split_chk.isChecked():
                src_avg[0] = +target_flux / len(src_arrs)
                snk_avg[0] = -target_flux / len(snk_arrs)
                mode_label = 'equal-split'
            else:
                sum_src = sum(arr[0].real for arr in src_arrs)
                if abs(sum_src) > 1e-12:
                    scale = target_flux / abs(sum_src)
                    src_avg = scale * src_avg
                    snk_avg = scale * snk_avg
                mode_label = 'measured-ratio'
            # Reconstruct Q(t) for both types.
            y_A = np.full_like(t, src_avg[0].real, dtype=float)
            y_V = np.full_like(t, snk_avg[0].real, dtype=float)
            kmax = min(len(src_avg), 4)
            for k in range(1, kmax):
                y_A += (src_avg[k].real * np.cos(k * omega * t)
                        - src_avg[k].imag * np.sin(k * omega * t))
                y_V += (snk_avg[k].real * np.cos(k * omega * t)
                        - snk_avg[k].imag * np.sin(k * omega * t))
            title = (f"MEASURED (per-type avg, {mode_label})   "
                     f"Q̄_A = {abs(src_avg[0].real):.3f},  "
                     f"Q̄_V = {abs(snk_avg[0].real):.3f} nL/s")

        # Display convention: flip each trace so its DC reads positive.
        # "Positive" = the sign that yields DC > 0 for that side, so
        # both A and V sit on the same axis and the magnitudes are
        # directly comparable.  Solver-internal sign convention (sinks
        # store -Q_outflow) is unaffected — this is purely cosmetic.
        s_A = -1.0 if float(np.nanmean(y_A)) < 0 else 1.0
        s_V = -1.0 if float(np.nanmean(y_V)) < 0 else 1.0
        y_A = y_A * s_A
        y_V = y_V * s_V

        ax.plot(t * 1000.0, y_A, color='C0', lw=1.5, label='A')
        ax.plot(t * 1000.0, y_V, color='C1', lw=1.5, label='V')
        ax.axhline(0, color='#888', lw=0.5)
        ax.set_xlabel('t (ms)', fontsize=8)
        ax.set_ylabel('Q(t) (nL/s)', fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        self._sim_custom_canvas.draw_idle()
