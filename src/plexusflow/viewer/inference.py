"""Inference tab (per-tile distensibility inference) and the radial-uncertain conductance fit.

Mixin of plexusflow.viewer.app.ReadOnlyMosaicViewer (methods moved verbatim from app.py)."""
from __future__ import annotations

from ._common import (  # noqa: F401
    List,
    MeasurementView,
    mcolors,
    np,
)


class InferenceMixin:

    # ── Simulate tab + colour-bar UI ───────────────────────────────────
    def _build_uncertain_tab(self, layout):
        """Fit the near-arterial uncertain-edge conductances (flow splitters)
        + ΔP so the network's downstream DC flows better match measured
        mean_Q, with a ΔP-only control.  Colours splitters by the fitted
        conductance correction and reports the improvement + predicted ΔP."""
        from qtpy.QtWidgets import (QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
                                    QComboBox, QHBoxLayout, QCheckBox, QWidget)
        self._unc_theta = {}          # frozenset(u,v) -> fitted θ multiplier
        self._unc_result = None
        intro = QLabel(
            "<b>Uncertain-vessel flow routing.</b>  Uncertain edges near the "
            "arterial inlet act as flow splitters with uncertain conductance. "
            "This fits their relative conductances (+ a free ΔP) so the "
            "network's <i>downstream</i> DC flows better match measured mean_Q — "
            "with a ΔP-only control isolating the splitters' effect.")
        intro.setWordWrap(True); layout.addWidget(intro)

        def _row(lbl, w):
            r = QWidget(); rl = QHBoxLayout(r); rl.setContentsMargins(0, 2, 0, 2)
            rl.addWidget(QLabel(lbl)); rl.addWidget(w); rl.addStretch(1)
            layout.addWidget(r); return w

        self._unc_tile = QComboBox()
        self._unc_tile.addItem("Whole plexus (best-tile mean_Q)", None)
        for t in sorted((self.G.graph.get('tile_f0_piv', {}) or {})):
            self._unc_tile.addItem(f"Tile {t}", int(t))
        _row("Match:", self._unc_tile)
        self._unc_dp = QDoubleSpinBox(); self._unc_dp.setRange(0.1, 500.); self._unc_dp.setValue(10.)
        self._unc_dp.setToolTip("Baseline ΔP (Pa); the fit scales it (free ΔP).")
        _row("ΔP baseline (Pa):", self._unc_dp)
        self._unc_unc_hops = QSpinBox(); self._unc_unc_hops.setRange(1, 60); self._unc_unc_hops.setValue(15)
        _row("Splitter reach (hops from source):", self._unc_unc_hops)
        self._unc_down_hops = QSpinBox(); self._unc_down_hops.setRange(1, 60); self._unc_down_hops.setValue(15)
        _row("Downstream targets beyond (hops):", self._unc_down_hops)
        self._unc_min_snr = QDoubleSpinBox(); self._unc_min_snr.setRange(-20., 40.); self._unc_min_snr.setValue(3.)
        _row("Min DC SNR (dB):", self._unc_min_snr)
        self._unc_reg = QDoubleSpinBox(); self._unc_reg.setRange(0., 5.); self._unc_reg.setDecimals(2)
        self._unc_reg.setValue(0.3); self._unc_reg.setToolTip("Keeps θ near 1; raise if θ rail.")
        _row("Regularisation:", self._unc_reg)

        self._unc_run_btn = QPushButton("Run fit")
        self._unc_run_btn.clicked.connect(self._run_uncertain_fit)
        layout.addWidget(self._unc_run_btn)
        self._unc_status = QLabel(""); self._unc_status.setWordWrap(True)
        layout.addWidget(self._unc_status)
        self._unc_result_lbl = QLabel(""); self._unc_result_lbl.setWordWrap(True)
        self._unc_result_lbl.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self._unc_result_lbl)
        self._unc_show_theta = QCheckBox("Colour splitters by θ (red ↑ / blue ↓ vs baseline)")
        self._unc_show_theta.setChecked(True)
        self._unc_show_theta.toggled.connect(lambda _v: self._refresh_edges())
        layout.addWidget(self._unc_show_theta)
        self._unc_plot_btn = QPushButton("Show plots (scatter / θ histogram)")
        self._unc_plot_btn.clicked.connect(self._show_uncertain_plots)
        self._unc_plot_btn.setEnabled(False)
        layout.addWidget(self._unc_plot_btn)
        self._unc_map_btn = QPushButton("Show residual map (naive vs corrected, spatial)")
        self._unc_map_btn.clicked.connect(self._show_uncertain_residual_map)
        self._unc_map_btn.setEnabled(False)
        layout.addWidget(self._unc_map_btn)


    def _run_uncertain_fit(self):
        from qtpy.QtWidgets import QApplication
        import time
        try:
            from scripts.fit_uncertain_conductances import fit_downstream
        except Exception:
            import importlib.util
            from pathlib import Path
            sp = Path(__file__).resolve().parents[2] / "scripts" / "fit_uncertain_conductances.py"
            spec = importlib.util.spec_from_file_location("fit_unc_mod", sp)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            fit_downstream = mod.fit_downstream
        self._unc_run_btn.setEnabled(False)
        self._unc_status.setText("<small>Running forward solve + fit … (~30–60 s)</small>")
        QApplication.processEvents()
        t0 = time.time()
        try:
            res = fit_downstream(
                self.G,
                unc_hops=int(self._unc_unc_hops.value()),
                down_hops=int(self._unc_down_hops.value()),
                min_snr_dc=float(self._unc_min_snr.value()),
                reg=float(self._unc_reg.value()),
                dp=float(self._unc_dp.value()),
                tile=self._unc_tile.currentData())
        except Exception as e:
            self._unc_status.setText(f"<small><font color='#c33'>Fit failed: {e}</font></small>")
            self._unc_run_btn.setEnabled(True); return
        self._unc_result = res
        self._unc_theta = {frozenset(uv): float(th)
                           for uv, th in zip(res['unc_edge_uv'], res['theta'])}
        if not res['n_targets'] or not len(res['theta']):
            self._unc_status.setText("<small><font color='#c33'>No splitters/targets — "
                                     "loosen the hop / SNR thresholds.</font></small>")
            self._unc_run_btn.setEnabled(True); return
        imp = 100*(res['r_control']-res['r_full'])/res['r_control']
        th = res['theta']
        self._unc_result_lbl.setText(
            f"splitters: {res['n_splitters']}   downstream targets: {res['n_targets']}\n"
            f"UNCORRECTED (ΔP only)  residual={res['r_control']:.3f}  ΔP={res['dp_control']:.1f} Pa\n"
            f"CORRECTED   (ΔP + θ)   residual={res['r_full']:.3f}  ΔP={res['dp_full']:.1f} Pa\n"
            f"→ downstream matching improved {imp:+.1f}%\n"
            f"θ: median {np.median(th):.2f}, range [{th.min():.2f}, {th.max():.2f}], "
            f"{int(np.sum(np.abs(np.log(th))>0.3))} moved >2×")
        self._unc_status.setText(f"<small>Done in {time.time()-t0:.0f} s.  "
                                 "Splitters coloured on the mosaic by θ.</small>")
        self._unc_plot_btn.setEnabled(True)
        self._unc_map_btn.setEnabled(True)
        self._unc_run_btn.setEnabled(True)
        self._refresh_edges()


    def _show_uncertain_plots(self):
        res = getattr(self, '_unc_result', None)
        if res is None or not len(res.get('theta', [])):
            return
        import matplotlib.pyplot as plt
        theta = res['theta']; tq = np.asarray(res['tgt_q'])
        sc, sf = np.asarray(res['sim_control']), np.asarray(res['sim_full'])
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        lim = np.percentile(np.abs(tq), 99) if len(tq) else 1.0
        for a, (qs, ttl, rr) in zip(ax[:2],
                [(sc, 'Uncorrected (ΔP only)', res['r_control']),
                 (sf, 'Corrected (ΔP + splitters)', res['r_full'])]):
            a.scatter(tq, qs, s=7, alpha=.25, edgecolors='none')
            a.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
            a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect('equal')
            a.set_xlabel('measured mean_Q (nL/s)'); a.set_title(f'{ttl}\nresidual = {rr:.3f}')
        ax[0].set_ylabel('simulated DC flow (nL/s)')
        ax[2].hist(np.log10(theta), bins=35, color='steelblue'); ax[2].axvline(0, color='k', ls='--')
        ax[2].set_xlabel('log10 θ'); ax[2].set_title('splitter conductance corrections')
        fig.suptitle(f"Uncertain-vessel fit — downstream matching "
                     f"{100*(res['r_control']-res['r_full'])/res['r_control']:+.1f}%,  "
                     f"predicted ΔP = {res['dp_full']:.1f} Pa")
        fig.tight_layout(); plt.show()


    def _show_uncertain_residual_map(self):
        """Spatial map of the per-edge downstream residual |sim − meas|:
        naive (ΔP only) | corrected (ΔP + splitters) | per-edge improvement.
        Shows WHERE the conductance fit helps (or hurts), on the mosaic."""
        res = getattr(self, '_unc_result', None)
        if res is None or not len(res.get('theta', [])) or not res.get('n_targets'):
            return
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        tq = np.asarray(res['tgt_q'])
        r_naive = np.abs(np.asarray(res['sim_control']) - tq)
        r_corr  = np.abs(np.asarray(res['sim_full'])    - tq)
        improve = r_naive - r_corr                      # + = better after fit
        pos = {n: (self.G.nodes[n]['x'], self.G.nodes[n]['y']) for n in self.G.nodes()}
        segs = np.array([[pos[u], pos[v]] for u, v in res['tgt_edge_uv']])
        usegs = np.array([[pos[u], pos[v]] for u, v in res['unc_edge_uv']])
        src = [n for n, a in self.G.nodes(data=True) if a.get('boundary_type') == 'source']
        snk = [n for n, a in self.G.nodes(data=True) if a.get('boundary_type') == 'sink']

        fig, ax = plt.subplots(1, 3, figsize=(19, 7), sharex=True, sharey=True)
        vmax = np.percentile(r_naive, 95) or 1.0
        for a, (rr, ttl) in zip(ax[:2],
                [(r_naive, f"Naive (ΔP only)   ‖r‖={res['r_control']:.3f}"),
                 (r_corr,  f"Corrected (ΔP+θ)  ‖r‖={res['r_full']:.3f}")]):
            lc = LineCollection(segs, array=rr, cmap='viridis',
                                norm=mcolors.Normalize(0, vmax), linewidths=1.4)
            a.add_collection(lc); plt.colorbar(lc, ax=a, shrink=.6, label='|sim − meas| (nL/s)')
            a.set_title(ttl)
        vlim = np.percentile(np.abs(improve), 95) or 1.0
        lci = LineCollection(segs, array=improve, cmap='RdBu',
                             norm=mcolors.TwoSlopeNorm(vcenter=0., vmin=-vlim, vmax=vlim),
                             linewidths=1.6)
        ax[2].add_collection(lci); plt.colorbar(lci, ax=ax[2], shrink=.6,
                                                label='|r_naive| − |r_corr|  (red=better)')
        ax[2].set_title('Per-edge improvement')
        for a in ax:                                     # splitters (grey) + boundary
            a.add_collection(LineCollection(usegs, colors='0.6', linewidths=0.8, zorder=1))
            for n in src: a.scatter(*pos[n], c='red', marker='^', s=70, zorder=5)
            for n in snk: a.scatter(*pos[n], c='blue', marker='v', s=70, zorder=5)
            a.autoscale(); a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"Downstream residual map — overall "
                     f"{100*(res['r_control']-res['r_full'])/res['r_control']:+.1f}% "
                     f"({res['n_targets']} target edges; grey = splitters)")
        fig.tight_layout(); plt.show()


    # ══════════════════════════════════════════════════════════════════
    # Inference tab — self-contained adapter region
    # ──────────────────────────────────────────────────────────────────
    # All inference plumbing (tile carve, inversion call, diagnostic
    # render) lives in the methods below.  To swap out the underlying
    # inversion algorithm, edit only:
    #   • `_inference_get_carve_edges`  (which edges form the carve)
    #   • `_inference_run`              (the per-tile call returning a
    #                                    dict of summary scalars)
    #   • `_inference_render_diagnostic` (the matplotlib popup figure)
    # The UI builder + dispatch glue below depend only on those
    # adapter methods, so the View / Time-series / Simulate tabs
    # don't need to change when the inversion model changes.
    # ══════════════════════════════════════════════════════════════════

    def _build_inference_tab(self, layout):
        """UI for the additive empirical-σ Bayesian inference.

        Per-edge measurement variance from the canonical Q_H1_snr_db
        field (r_j = 10^(-SNR/10)).  Total Σ_j = |Q_H1,j|²·(r_j + s²)
        where s² is the squared fractional model error, marginalized
        analytically over a 21-pt log grid.  D₀ has a log-normal prior
        centred at 1.5×10⁻³ /Pa, τ = ln 10.

        Per-edge D scaling:  D_e = D₀ · (R_e / R_ref)^α.  α = 0
        is the default (uniform D₀); α = +1 lets bigger vessels be
        more compliant (Murray-inverse, common for embryonic walls).
        D̂ is reported AT R_ref (median measured-edge radius).

        After running on a tile, click an edge in the viewer (the
        Time-series tab updates "Selected edge") then hit the Compare
        button to see measured vs predicted Q(t) for that edge."""
        from qtpy.QtWidgets import (
            QLabel, QPushButton, QGroupBox, QHBoxLayout, QVBoxLayout,
            QComboBox, QCheckBox, QTextEdit, QDoubleSpinBox,
        )
        # ── Tile picker ──
        pick_group = QGroupBox("Tile")
        pick_lay = QVBoxLayout(pick_group)
        self._inference_tile_ids = self._inference_available_tiles()
        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel("Tile ID:"))
        self._inf_tile_combo = QComboBox()
        for tid in self._inference_tile_ids:
            self._inf_tile_combo.addItem(str(tid), userData=int(tid))
        if not self._inference_tile_ids:
            self._inf_tile_combo.addItem("(none — no PIV tiles found)")
            self._inf_tile_combo.setEnabled(False)
        self._inf_tile_combo.currentIndexChanged.connect(
            self._on_inference_tile_change)
        pick_row.addWidget(self._inf_tile_combo)
        pick_row.addStretch(1)
        pick_lay.addLayout(pick_row)
        self._inf_show_carve_chk = QCheckBox("Highlight carve on mosaic")
        self._inf_show_carve_chk.setChecked(False)
        self._inf_show_carve_chk.toggled.connect(
            self._on_inference_show_carve_toggle)
        pick_lay.addWidget(self._inf_show_carve_chk)
        layout.addWidget(pick_group)

        # ── Parameters ──
        param_group = QGroupBox("Priors and model")
        param_lay = QVBoxLayout(param_group)
        sb_row = QHBoxLayout()
        sb_row.addWidget(QLabel("σ_b (H₁ boundary forcing, Pa):"))
        self._inf_sigma_b_spin = QDoubleSpinBox()
        self._inf_sigma_b_spin.setRange(0.1, 1000.0)
        self._inf_sigma_b_spin.setDecimals(1)
        self._inf_sigma_b_spin.setSingleStep(1.0)
        self._inf_sigma_b_spin.setValue(7.0)
        self._inf_sigma_b_spin.setToolTip(
            "Absolute b prior σ (independent of s²).  Sensitivity\n"
            "sweep showed D̂ is invariant across [1, 100] Pa on\n"
            "identified tiles.  Check 'flat' to use σ_b = 10⁴ Pa,\n"
            "the wide-prior limit (formally improper but numerically\n"
            "stable — sees the b̂ that the data alone can support).")
        sb_row.addWidget(self._inf_sigma_b_spin)
        self._inf_flat_b_prior_cb = QCheckBox("flat (σ_b → ∞)")
        self._inf_flat_b_prior_cb.setToolTip(
            "When checked, override σ_b with 10⁴ Pa — far above the\n"
            "BC confound threshold (~50 Pa) — to approximate an\n"
            "improper flat prior on b_H₁.  Use this to confirm that\n"
            "the data alone (without a prior on b magnitude) push\n"
            "D̂ toward the resistive limit, exposing the D·|b̂|\n"
            "structural identifiability ridge.")
        self._inf_flat_b_prior_cb.toggled.connect(
            lambda checked: self._inf_sigma_b_spin.setEnabled(not checked))
        sb_row.addWidget(self._inf_flat_b_prior_cb)
        sb_row.addStretch(1)
        param_lay.addLayout(sb_row)
        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("α  (D_e = D₀·(R/R_ref)^α):"))
        self._inf_alpha_spin = QDoubleSpinBox()
        self._inf_alpha_spin.setRange(-3.0, 3.0)
        self._inf_alpha_spin.setDecimals(2)
        self._inf_alpha_spin.setSingleStep(0.25)
        self._inf_alpha_spin.setValue(0.0)
        self._inf_alpha_spin.setToolTip(
            "Radius-scaling of per-edge D.\n"
            "  α = 0    → uniform D₀  (default)\n"
            "  α = +1   → D linear in R (bigger vessels more compliant)\n"
            "  α = −1   → D inverse in R (Murray-like, stiffer big vessels)\n"
            "R_ref = median measured-edge radius for the tile.\n"
            "D̂ is reported at R = R_ref so it's interpretable as the\n"
            "typical-vessel distensibility independent of α.")
        alpha_row.addWidget(self._inf_alpha_spin)
        alpha_row.addStretch(1)
        param_lay.addLayout(alpha_row)
        # ── BC correlation length ℓ (spatial smoothness on b̂_H₁) ──
        bcL_row = QHBoxLayout()
        bcL_row.addWidget(QLabel("BC correlation length ℓ (px):"))
        self._inf_bc_length_spin = QDoubleSpinBox()
        self._inf_bc_length_spin.setRange(0.0, 5000.0)
        self._inf_bc_length_spin.setDecimals(0)
        self._inf_bc_length_spin.setSingleStep(50.0)
        self._inf_bc_length_spin.setValue(0.0)
        self._inf_bc_length_spin.setToolTip(
            "Spatial smoothness prior on b_H₁ across boundary nodes.\n"
            "    C_ij = σ_b² · exp(−d_ij² / 2ℓ²)\n"
            "  ℓ = 0    → independent BCs (default; no smoothing)\n"
            "  ℓ ≈ tile size → adjacent BCs correlated; less BC freedom\n"
            "                  → harder for boundary forcing to absorb\n"
            "                    D-dependence (constrains b̂_H₁ → tighter D̂)\n"
            "Use this to test whether a tile's D̂ is driven by BC\n"
            "absorption flexibility — if D̂ shifts substantially as ℓ\n"
            "grows, the original tile was boundary-absorption-dominated.")
        bcL_row.addWidget(self._inf_bc_length_spin)
        bcL_row.addStretch(1)
        param_lay.addLayout(bcL_row)
        # ── Harmonic-mode selector ──
        # Use H₁ only (default), H₂ only, or both jointly.  In joint mode
        # b_H₁ and b_H₂ are independent (each gets its own Gaussian prior)
        # but D is shared, so H₂ adds an independent constraint on D.
        from PyQt5.QtWidgets import QComboBox
        harm_row = QHBoxLayout()
        harm_row.addWidget(QLabel("Harmonic:"))
        self._inf_harmonic_combo = QComboBox()
        self._inf_harmonic_combo.addItems(["H₁ only", "H₂ only", "H₁ + H₂ joint"])
        self._inf_harmonic_combo.setCurrentIndex(0)
        self._inf_harmonic_combo.setToolTip(
            "Which harmonic(s) the inference uses.\n"
            "  H₁ only      — default, single channel at f₀.\n"
            "  H₂ only      — single channel at 2f₀ (SNR_H₂ usually lower).\n"
            "  H₁ + H₂ joint — both channels, b̂_H₁ and b̂_H₂ each get\n"
            "                   their own prior, D shared.  Tightens CIs\n"
            "                   ~30-40% on tiles with adequate H₂ SNR;\n"
            "                   does NOT break the structural b–D ridge.")
        harm_row.addWidget(self._inf_harmonic_combo)
        harm_row.addStretch(1)
        param_lay.addLayout(harm_row)
        # ── SNR-based σ floor (opt-in hybrid noise model) ──
        from PyQt5.QtWidgets import QCheckBox
        snr_row = QHBoxLayout()
        self._inf_snr_floor_cb = QCheckBox("Use SNR-based σ (replace empirical)")
        self._inf_snr_floor_cb.setChecked(False)
        self._inf_snr_floor_cb.setToolTip(
            "Off (default): noise model is empirical "
            "σ²_j = â² + ŝ²·|Q_j|², with (â, ŝ) marginalised.\n"
            "On: REPLACE the empirical fit with the SNR-implied "
            "σ²_j = |Q_j|²·10^(−SNR_j/10) per edge, using the stored "
            "SNR (Q_H1_snr_db on canonical, harmonics[k].snr_db on v2).\n"
            "Trusts the stored SNR completely.  Tightens D̂ when "
            "the empirical fit is over-inflating ŝ to absorb model "
            "misfit (RMS|z| ≪ 1), but vulnerable to misleading "
            "stored SNR (e.g. embryo-motion-contaminated edges).")
        snr_row.addWidget(self._inf_snr_floor_cb)
        snr_row.addStretch(1)
        param_lay.addLayout(snr_row)
        info = QLabel(
            "<small><i>σ²_j = â² + ŝ²·|Q_H,j|² "
            "(additive + proportional, both marginalised).<br>"
            "γ (SNR weighting) → 0 from inferred-shape test; "
            "SNR field NOT used.<br>"
            "D₀ ~ LogNormal(ln 1.5×10⁻³, ln 10).  "
            "Anatomical A/V → BC by default.</i></small>")
        info.setWordWrap(True)
        param_lay.addWidget(info)
        layout.addWidget(param_group)

        # ── Actions ──
        act_row = QHBoxLayout()
        self._inf_run_btn = QPushButton("Run inference")
        self._inf_run_btn.clicked.connect(self._on_inference_run_clicked)
        act_row.addWidget(self._inf_run_btn)
        self._inf_diag_btn = QPushButton("D₀ posterior")
        self._inf_diag_btn.setEnabled(False)
        self._inf_diag_btn.clicked.connect(
            self._on_inference_diagnostic_clicked)
        act_row.addWidget(self._inf_diag_btn)
        layout.addLayout(act_row)
        compare_row = QHBoxLayout()
        self._inf_compare_btn = QPushButton(
            "Compare measured vs predicted Q(t) (selected edge)")
        self._inf_compare_btn.setEnabled(False)
        self._inf_compare_btn.setToolTip(
            "After running inference, click an edge in the viewer\n"
            "(Time-series tab shows the selected edge), then click\n"
            "this button.  Pops up a comparison of measured Q(t) and\n"
            "the inference's predicted Q(t) at MAP D₀ and MAP b̂.")
        self._inf_compare_btn.clicked.connect(
            self._on_inference_compare_clicked)
        compare_row.addWidget(self._inf_compare_btn)
        layout.addLayout(compare_row)

        # ── Results ──
        self._inf_results_box = QTextEdit()
        self._inf_results_box.setReadOnly(True)
        self._inf_results_box.setMaximumHeight(220)
        self._inf_results_box.setPlaceholderText(
            "Pick a tile and click 'Run inversion'.")
        layout.addWidget(self._inf_results_box)
        # Status
        self._inf_status_lbl = QLabel(
            "<small><i>Inference results live in memory; nothing is "
            "written back to the gpickle.</i></small>")
        self._inf_status_lbl.setWordWrap(True)
        layout.addWidget(self._inf_status_lbl)

        # Carve overlay layer slots (created lazily on first toggle).
        self._inference_carve_layer = None        # edge Shapes layer
        self._inference_carve_node_layer = None   # boundary-node Points layer
        self._inference_last_result = None
        # OF region overlay on the mosaic — curved orange ribbon showing
        # the actual integration region for the last OF run.  Persists
        # until the next OF run replaces it or "Clear OF region" is hit.
        self._of_region_layer = None

        # No initial carve render — the checkbox defaults to off so
        # the launch view stays uncluttered.  The first toggle-on
        # triggers `_refresh_inference_carve_overlay` lazily.

    # ── Adapter ──────────────────────────────────────────────────────
    # These three methods are the ONLY entry points into the inversion
    # algorithm.  Swap them out (or replace the whole region) to change
    # what "Run inversion" means.
    def _inference_available_tiles(self) -> List[int]:
        """Return a sorted list of tile IDs the inversion can run on.
        Default rule: any tile that appears as `tile_id` on at least
        one usable PIV measurement."""
        seen = set()
        for _, _, d in self.G.edges(data=True):
            for m in (d.get('measurements_piv') or []):
                # v4 migration: was m.get('tile_id')
                tid = MeasurementView(m).tile_id
                if tid is None:
                    continue
                try:
                    seen.add(int(tid))
                except (TypeError, ValueError):
                    pass
        return sorted(seen)


    def _inference_get_carve_topology(self, tile_id: int) -> dict:
        """Return carve geometry for the given tile as a dict with keys
        `edges` (list of (u,v)) and `boundary_nodes` (list of node ids).
        Wraps `scripts.inspect_tile.build_tile_problem`."""
        try:
            import sys as _sys
            from pathlib import Path as _Path
            _scripts = _Path(self.graph_path).resolve().parent
            # Walk up to find the project root with a `scripts/` folder.
            for _ancestor in [_scripts] + list(_scripts.parents):
                if (_ancestor / 'scripts' / 'inspect_tile.py').exists():
                    _root = _ancestor
                    break
            else:
                from pertile.viewer import mosaic_readonly_app as _self_mod
                _root = _Path(_self_mod.__file__).resolve().parents[2]
            if str(_root) not in _sys.path:
                _sys.path.insert(0, str(_root))
            if str(_root / 'scripts') not in _sys.path:
                _sys.path.insert(0, str(_root / 'scripts'))
            from inspect_tile import build_tile_problem
            prob = build_tile_problem(self.G, int(tile_id))
        except Exception as e:
            print(f"  inference: carve extraction failed ({type(e).__name__}: {e})")
            return {'edges': [], 'boundary_nodes': []}
        return {
            'edges': [tuple(e) for e in prob.get('edges_in', [])],
            'boundary_nodes': list(prob.get('boundary_nodes', [])),
        }


    def _inference_run(self, tile_id: int, sigma_b_Pa: float = 7.0,
                        alpha: float = 0.0, bc_length_px: float = 0.0,
                        harmonic_mode: str = 'H1',
                        cap_omega_mult: float = None,
                        use_snr_floor: bool = False):
        """Run the additive empirical-σ Bayesian inference on one tile.

        Model:
          Σ_j = |Q_H1,j|² · (r_j + s²)        r_j = 10^(−SNR_H1,j/10)
          D_e = D₀ · (R_e / R_ref)^α          R_ref = median measured-edge R
          b_H1 ~ N(0, σ_b² · I)
          D₀ ~ LogNormal(ln 1.5×10⁻³, ln 10)

        s² is marginalised analytically over a 21-pt log grid in
        [10⁻⁴, 10⁰].  b_H1 is marginalised analytically (Gaussian
        conjugate).  D₀ posterior is on a 41-pt log grid in
        [10⁻⁵, 10⁻¹].

        Returns a dict with:
          tile_id, alpha, sigma_b
          D_hat, lo, hi, width, tier
          s2_at_map, frac_err_pct
          log_p_D (1D D₀ posterior, normalised)
          D_GRID
          n_meas, snr_median, R_ref_um
          b_hat (MAP b at MAP D₀, MAP s²) — real-imag stacked, length 2·n_B
          Q_H1_pred (per-edge predicted H1 phasor at MAP, all edges)
          prob (the carve)
          R_e (per-edge radius array, in metres)
        """
        import math
        import networkx as nx
        from scipy.special import logsumexp
        from pertile.analysis.local_pressure_inference import (
            _edge_geometry, extract_tile_subgraph_spatial,
        )
        # Fixed physical constants — match the standalone sweep scripts.
        # Viscosity (Pa·s), cardiac f₀ (Hz, tile-anchored), pixel size (m/px).
        MU = 3.5e-3
        PX_SIZE_M = 1.7e-6
        # Per-tile f₀: prefer the tile-anchored value, fall back to the
        # graph-level default, then 2.5 Hz.
        tile_f0_map = self.G.graph.get('tile_f0_piv') or {}
        F0_HZ = float(tile_f0_map.get(int(tile_id)) or
                       self.G.graph.get('f0_hz') or 2.5)
        # ── Carve the tile's spatial neighbourhood ──
        try:
            edges_in, _, boundary_nodes, interior_nodes = (
                extract_tile_subgraph_spatial(
                    self.G, int(tile_id),
                    inset_frac=0.0,
                    restrict_to_tile_piv_nodes=False,
                    drop_dangling_boundaries=True))
        except Exception as e:
            return {'error': f'spatial carve failed: {e}'}
        edges_in = [tuple(e) for e in edges_in]
        boundary_nodes = list(boundary_nodes)
        interior_nodes = list(interior_nodes)
        # ── Promote anatomical A/V nodes to boundary ──
        # The spatial carve defines "boundary" by perimeter geometry, but
        # nodes with boundary_type == 'source' (A) or 'sink' (V) are
        # anatomical pressure anchors (heart-side inflow / venous return).
        # They MUST be treated as open boundaries; left as interior, the
        # inference would force the model to equilibrate at them, which
        # is physically wrong.  Promote any A/V node currently in the
        # carve interior to boundary so its pressure is a free parameter
        # of the inference, not a quantity the network has to balance.
        try:
            _av_in_interior = [
                n for n in interior_nodes
                if self.G.nodes[n].get('boundary_type') in ('source', 'sink')]
            if _av_in_interior:
                _av_set = set(_av_in_interior)
                boundary_nodes = boundary_nodes + _av_in_interior
                interior_nodes = [n for n in interior_nodes
                                    if n not in _av_set]
            _n_av_promoted = len(_av_in_interior)
        except Exception:
            _n_av_promoted = 0
        # ── Build the subgraph (carries radius/length attrs needed below) ──
        sub = nx.Graph()
        for n in set(boundary_nodes) | set(interior_nodes):
            nd = self.G.nodes[n]
            sub.add_node(n, x=float(nd.get('x', 0.0)),
                          y=float(nd.get('y', 0.0)))
        for (u, v) in edges_in:
            sub.add_edge(u, v, **dict(self.G.edges[u, v]))
        # ── Build per-edge Q observations from canonical TOP-LEVEL fields ──
        # Matches Steven's build_tile_problem convention: each edge gets
        # its canonical-best Q_DC / Q_H1 (promoted from the best-tile
        # measurement during graph build), regardless of whether the
        # source tile equals `tile_id`.  This is what /tmp/sweep_additive.py
        # used and is the basis of the headline D̂ numbers.  Per-tile
        # measurements (m['tile_id'] == tile_id specifically) are used
        # only by the comparison plot.
        # ── Per-edge observations: TILE-LOCAL matched pairs ──
        # Read Q_DC, Q_H1, Q_DC_SNR, Q_H1_SNR all from the SAME per-tile
        # entry in measurements_piv (matched on tile_id).  An edge that
        # was *not* measured on this tile drops out of the likelihood
        # entirely — no silent borrowing from the canonical top-level
        # field (which may have been promoted from a different tile and
        # would carry that tile's amplitude / phase calibration).
        #
        # This is the per-tile inference's defining commitment: every
        # (Q, σ) pair in y comes from THIS tile's own acquisition.
        # Bridges and edges only measured on other tiles still appear in
        # the forward-model topology (so the carve geometry is unchanged)
        # but contribute nothing to y.
        nL_PER_M3 = 1e12
        n_edges_in = len(edges_in)
        Q_DC_obs = np.full(n_edges_in, np.nan, dtype=float)
        Q_H1_obs = np.full(n_edges_in, np.nan, dtype=complex)
        Q_H2_obs = np.full(n_edges_in, np.nan, dtype=complex)
        valid_dc = np.zeros(n_edges_in, dtype=bool)
        valid_h1 = np.zeros(n_edges_in, dtype=bool)
        valid_h2 = np.zeros(n_edges_in, dtype=bool)
        snr_per_edge = np.full(n_edges_in, np.nan)
        snr_H2_per_edge = np.full(n_edges_in, np.nan)
        snr_DC_per_edge = np.full(n_edges_in, np.nan)
        # Per-edge time-series length N — needed to convert the
        # canonical SNR_H1_db (a time-series scale derived from the
        # band-limited residual MS_BL) into a phasor estimation variance.
        # See σ²_meas below for the relationship.
        N_per_edge = np.full(n_edges_in, np.nan)
        for i, (u, v) in enumerate(edges_in):
            ed = self.G.edges[u, v]
            # Find THIS tile's measurement on this edge.  None → edge
            # is unmeasured on this tile (a bridge or only-measured-
            # elsewhere); skip from the likelihood.
            m_tile = next((mm for mm in (ed.get('measurements_piv') or [])
                            if MeasurementView(mm).tile_id == int(tile_id)), None)
            if m_tile is None:
                continue
            mv_tile = MeasurementView(m_tile)
            # Sign convention: prefer the per-tile measurement's own
            # flow direction.  v2 edges have flow_from/flow_to set ONLY
            # on per-tile entries (top-level is None until promotion
            # writes flow_from_piv / flow_to_piv).  Canonical edges
            # historically had them at the top level too — fall back
            # there if the per-tile dict doesn't carry them.
            # v4 migration: was m_tile.get('flow_from') / .get('flow_to')
            ff = mv_tile.flow_from
            ft = mv_tile.flow_to
            if ff is None or ft is None:
                # v4 migration: ed.get('flow_from_piv') is dropped in v4.
                # The v4-clean fallback source for flow_from/_to on the edge
                # is MeasurementView.best(ed).flow_from/_to; keep the legacy
                # ed.get('flow_from_piv') tail as a v3-graph safety net.
                mv_best = MeasurementView.best(ed)
                if mv_best is not None:
                    ff = ed.get('flow_from', mv_best.flow_from)
                    ft = ed.get('flow_to',   mv_best.flow_to)
                else:
                    ff = ed.get('flow_from', ed.get('flow_from_piv'))
                    ft = ed.get('flow_to',   ed.get('flow_to_piv'))
            if ff is None or ft is None:
                continue
            sign = 1.0 if (ff == u and ft == v) else -1.0
            # DC channel.  Canonical: 'Q_DC'.  v2: 'mean_Q'.
            mq = m_tile.get('Q_DC')
            if mq is None or not np.isfinite(mq):
                # v4 migration: was m_tile.get('mean_Q')
                mq = mv_tile.harm.mean_Q
            if mq is not None and np.isfinite(mq):
                Q_DC_obs[i] = float(mq) * sign / nL_PER_M3
                valid_dc[i] = True
            # H1 / H2 channels.  Canonical: 'Q_H1_amp','Q_H1_phi','Q_H2_amp',...
            #                    v2:        harmonics[0]['amp']/['phi'], harmonics[1]...
            amp = m_tile.get('Q_H1_amp'); phase = m_tile.get('Q_H1_phi')
            amp2 = m_tile.get('Q_H2_amp'); phase2 = m_tile.get('Q_H2_phi')
            # v4 migration: was m_tile.get('harmonics')
            _harms = mv_tile.harm.harmonics or []
            if (amp is None or not np.isfinite(amp)) and len(_harms) >= 1:
                amp = _harms[0].get('amp')
                phase = _harms[0].get('phi')
            if (amp2 is None or not np.isfinite(amp2)) and len(_harms) >= 2:
                amp2 = _harms[1].get('amp')
                phase2 = _harms[1].get('phi')
            if (amp is not None and phase is not None
                    and np.isfinite(amp) and np.isfinite(phase)):
                Q_H1_obs[i] = (float(amp) * np.exp(1j * float(phase))
                                * sign / nL_PER_M3)
                valid_h1[i] = True
            if (amp2 is not None and phase2 is not None
                    and np.isfinite(amp2) and np.isfinite(phase2)):
                Q_H2_obs[i] = (float(amp2) * np.exp(1j * float(phase2))
                                * sign / nL_PER_M3)
                valid_h2[i] = True
            # SNRs (per-edge measurement noise via the additive model)
            # Try canonical schema first; fall back to v2 schema (harmonics[k].snr_db).
            snr_H1 = m_tile.get('Q_H1_snr_db', np.nan)
            snr_H2 = m_tile.get('Q_H2_snr_db', np.nan)
            snr_DC = m_tile.get('Q_DC_snr_db', np.nan)
            # v4 migration: was m_tile.get('harmonics')
            _harms = mv_tile.harm.harmonics or []
            if (snr_H1 is None or not np.isfinite(snr_H1)) and len(_harms) >= 1:
                snr_H1 = _harms[0].get('snr_db', np.nan)
            if (snr_H2 is None or not np.isfinite(snr_H2)) and len(_harms) >= 2:
                snr_H2 = _harms[1].get('snr_db', np.nan)
            snr_per_edge[i] = snr_H1
            snr_H2_per_edge[i] = snr_H2
            snr_DC_per_edge[i] = snr_DC
            Q_t = mv_tile.Q_t  # v4 migration: was m_tile.get('Q_t')
            if Q_t is not None:
                try:
                    N_per_edge[i] = len(Q_t)
                except TypeError:
                    pass
        prob = dict(
            sub=sub, edges_in=edges_in,
            boundary_nodes=boundary_nodes, interior_nodes=interior_nodes,
            Q_DC_obs=Q_DC_obs, Q_H1_obs=Q_H1_obs, Q_H2_obs=Q_H2_obs,
            valid_dc=valid_dc, valid_h1=valid_h1, valid_h2=valid_h2,
        )
        # Choose validity mask based on the harmonic mode.
        # H1-only and H1+H2 both require valid_h1; H2-only requires valid_h2.
        if harmonic_mode == 'H2':
            valid = prob['valid_dc'] & prob['valid_h2']
        elif harmonic_mode == 'H1+H2':
            # Joint mode requires both channels measured on the same edges.
            valid = prob['valid_dc'] & prob['valid_h1'] & prob['valid_h2']
        else:
            valid = prob['valid_dc'] & prob['valid_h1']
        n_v = int(valid.sum())
        if n_v < 10:
            return {'error': f'Tile {tile_id} has only {n_v} valid edges (< 10) under harmonic mode {harmonic_mode}'}
        # Per-edge R, L in meters
        edges_in = prob['edges_in']
        R_e = np.zeros(len(edges_in)); L_e = np.zeros(len(edges_in))
        for k, (u, v) in enumerate(edges_in):
            R_m, L_m = _edge_geometry(self.G.edges[u, v], PX_SIZE_M)
            R_e[k] = max(R_m, 1e-9); L_e[k] = max(L_m, 1e-9)
        R_ref = float(np.median(R_e[valid]))
        R_factor = (R_e / R_ref) ** float(alpha)
        # ── Per-edge measurement noise (phasor variance, white-residual) ──
        # Honest additive-plus-proportional noise model:
        #
        #     σ²_j  =  a²  +  s² · |Q_j|²
        #
        # a (m³/s)         — measurement-noise floor, signal-independent.
        #                    Keeps low-|Q| edges from running away under
        #                    pure proportional weighting (the σ²→0 problem
        #                    with σ²∝|Q|² alone).
        # s² (dimensionless) — proportional misfit (ridge + KCL deficit,
        #                    both scale with |Q|).
        #
        # Both inferred / marginalised — we no longer assert SNR-based
        # weighting (r_j) or the (2/N) phasor factor; the previous test
        # (learned_noise_model_full.py) showed γ → 0 on the SNR axis with
        # posterior mass 1.000 on every tile.  The canonical SNR field
        # carries no per-edge precision information, so dropping r_j is
        # the inferred preference, not an assumption.
        #
        # The crossover  |Q|* = a/s  is the flow magnitude below which
        # measurement noise dominates and above which model misfit
        # dominates — a real, interpretable, defensible number.
        # ── Choose data channel(s) for the likelihood ──
        # primary_channel = the channel we marginalise b for and use to
        # build the "main" T(D) etc.  In joint mode, H₁ is primary and
        # H₂ is added on top.
        primary_channel = 'H2' if harmonic_mode == 'H2' else 'H1'
        if primary_channel == 'H1':
            Q_h_primary = prob['Q_H1_obs'][valid]
        else:
            Q_h_primary = prob['Q_H2_obs'][valid]
        Q_abs2 = np.abs(Q_h_primary) ** 2
        Q_abs = np.sqrt(Q_abs2)
        y_H1 = np.concatenate([Q_h_primary.real, Q_h_primary.imag])
        valid_idx = np.where(valid)[0]
        Q_abs2_stack = np.concatenate([0.5 * Q_abs2, 0.5 * Q_abs2])
        # H₂ data on the same edges, used only when harmonic_mode == 'H1+H2'.
        if harmonic_mode == 'H1+H2':
            Q_h2_data = prob['Q_H2_obs'][valid]
            Q_abs2_H2 = np.abs(Q_h2_data) ** 2
            y_H2 = np.concatenate([Q_h2_data.real, Q_h2_data.imag])
        else:
            Q_h2_data = None; Q_abs2_H2 = None; y_H2 = None
        # ── Boundary forcing prior ──
        # Default (bc_length_px = 0): b_H₁,n ~ N(0, σ_b²) independently.
        #                              P₀ = σ_b² · I  (diagonal).
        # Spatial-smoothness mode (bc_length_px > 0): Gaussian-process
        # prior — adjacent BC nodes are correlated, so the inference
        # cannot place arbitrarily different b̂ values at nearby BCs to
        # absorb D-dependence.
        #
        #   C_ij = σ_b² · exp(−d_ij² / 2ℓ²)         (squared-exponential)
        #
        # with d_ij = Euclidean distance between BC nodes i, j (mosaic
        # px), and ℓ = bc_length_px the user-set correlation length.
        # A small ridge is added to guarantee numerical positive
        # definiteness when many BC nodes are spatially close.
        n_B = len(prob['boundary_nodes'])
        if bc_length_px and bc_length_px > 0:
            bc_xy = np.zeros((n_B, 2))
            for i_b, n in enumerate(prob['boundary_nodes']):
                nd = self.G.nodes[n]
                bc_xy[i_b] = (float(nd.get('x', 0.0)),
                               float(nd.get('y', 0.0)))
            d_sq = ((bc_xy[:, None, :] - bc_xy[None, :, :]) ** 2).sum(axis=2)
            C_bc = np.exp(-d_sq / (2.0 * float(bc_length_px) ** 2))
            C_bc = sigma_b_Pa ** 2 * C_bc
            C_bc += 1e-6 * sigma_b_Pa ** 2 * np.eye(n_B)
            # Real-form block-diagonal: Re and Im components are
            # independent under the complex circularly-symmetric Gaussian.
            zero_blk = np.zeros((n_B, n_B))
            P0 = 0.5 * np.block([[C_bc, zero_blk],
                                   [zero_blk, C_bc]])
        else:
            P0 = np.diag(np.concatenate([
                np.full(n_B, 0.5 * sigma_b_Pa ** 2),
                np.full(n_B, 0.5 * sigma_b_Pa ** 2),
            ]))
        # ── D₀ grid + physical-validity cap ──
        # The linear-RC forward model is only valid where the per-edge
        # compliance number  C_e = 8μω·L²·D / R²  ≪ 1.  Beyond C_e ~ 1,
        # the wall-storage current is no longer a perturbation, the
        # distributed/lumped agreement breaks down at the O(1) level,
        # and (more importantly) the linear wall law ΔA/A = D·ΔP is
        # itself questionable.
        #
        # Cap the D₀ grid at the value where  max_e(C_e) = C_E_CAP.
        # This is a tile-specific bound determined by edge geometry.
        # Without the cap, degenerate tiles can extrapolate their
        # broad posterior into the high-D tail (C_e ≳ 1) and
        # manufacture a "finite-compliance preference" from a regime
        # where the model itself is not trustworthy.  With the cap,
        # the inference is restricted to the physically defensible
        # window and degenerate tiles correctly read as unidentified
        # rather than borderline-strong.
        C_E_CAP = 0.3                              # physically-defensible upper bound
        # Per-edge geometry factor K_e = 8μω·L²/R² (so C_e = D·K_e).
        # Use ω at the channel that drives the largest κ²:
        #   H₁-only  → ω = 2π·f₀
        #   H₂-only  → ω = 4π·f₀                (κ² doubled per edge)
        #   joint    → use H₂'s ω to keep both channels safely inside C_e cap
        # Override: cap_omega_mult lets the caller force a *shared* D-grid
        # across H1-only and H2-only runs (set to 2.0 in both calls →
        # both grids anchored on the tighter H2 cap).  Without it, falls
        # back to the existing per-mode behavior.
        if cap_omega_mult is not None:
            omega_for_cap = float(cap_omega_mult) * 2.0 * np.pi * F0_HZ
        else:
            omega_for_cap = (4.0 * np.pi * F0_HZ
                              if harmonic_mode in ('H2', 'H1+H2')
                              else 2.0 * np.pi * F0_HZ)
        K_e = np.zeros(len(edges_in))
        valid_geom = (R_e > 0) & (L_e > 0)
        K_e[valid_geom] = (8.0 * MU * omega_for_cap
                            * L_e[valid_geom] ** 2
                            / R_e[valid_geom] ** 2)
        # Cap based on measured edges (α-scaling tweaks per-edge D, but
        # for the cap we use D₀ at R_ref → D_e = D₀; the α correction
        # is at most a factor R_max/R_ref which is order 1).
        K_meas_max = float(K_e[valid].max() if valid.any() else 0.0)
        D_cap = (C_E_CAP / K_meas_max) if K_meas_max > 0 else 1.0
        LOG_D_CAP = np.log10(max(D_cap, 1e-5))
        LOG_D_GRID = np.arange(-5.0, min(-1.0, LOG_D_CAP) + 0.001, 0.10)
        D_GRID = 10.0 ** LOG_D_GRID
        # ── Noise-model grids (additive a + proportional s²) ──
        # a (m³/s) is the measurement-noise floor; ŝ² is the proportional
        # misfit (dimensionless).  Grids trimmed to the support actually
        # populated by the 53/84-tile sweep posteriors:
        #   â ∈ [10⁻⁵, 10⁻² nL/s]  →  log₁₀(a/[m³/s]) ∈ [-17, -14]
        #   ŝ² ∈ [0.04, 4]          →  log₁₀(s²) ∈ [-1.4, 0.6]
        # Wide enough to not rail; small enough to keep per-tile inference
        # well under 10 s (was ~30 s when the grids spanned [-18, -13] ×
        # [-4, 0.4]).  Extend if you find a tile railing the boundary.
        LOG_A_GRID = np.arange(-17.0, -14.0 + 0.001, 0.30)   # 11 pts
        A_GRID = 10.0 ** LOG_A_GRID
        LOG_S2_GRID = np.arange(-1.4, 0.6 + 0.001, 0.30)     # 7 pts
        S2_GRID = 10.0 ** LOG_S2_GRID
        MU_D, TAU_D = np.log(1.5e-3), np.log(10.0)
        log_prior_D = -0.5 * ((np.log(D_GRID) - MU_D) / TAU_D) ** 2
        # ── O(C_e) distributed two-port forward model ──
        # The full distributed two-port is
        #   Y_self  = G · κ · coth(κ)
        #   Y_trans = G · κ · csch(κ)
        # with κ² = i·ω·D·8μ·L²/R² = i·C_e (Desktop's dimensionless number).
        # In the small-C_e regime (C_e ≲ 0.3, enforced by the D-grid cap
        # above) the hyperbolics are indistinguishable from their two-term
        # polynomial expansion to O(C_e²) ~ 0.003%:
        #
        #   ξ·coth(ξ) = 1 + ξ²/3 − ξ⁴/45 + …    →   Y_self  ≈ G · (1 + κ²/3)
        #   ξ·csch(ξ) = 1 − ξ²/6 + 7ξ⁴/360 − … →   Y_trans ≈ G · (1 − κ²/6)
        #
        # We use the polynomial form because (a) the regime of validity is
        # explicit in the code (anything past O(C_e²) is dropped, and the
        # D_cap keeps us there), (b) no small-|κ| special case is needed,
        # (c) it is the "right" linearization — the lumped-π misdistributes
        # the same total shunt as Y_self = G(1 + κ²/2), Y_trans = G, which
        # has the wrong O(C_e) coefficients per edge (a 0.6% error growing
        # linearly with D — exactly the bug that inflated high-D tails).
        #
        # Nodal admittance:    Y[u,u] += Y_self,e  ;  Y[u,v] -= Y_trans,e
        # Edge flow readout:   Q_e = Y_self,e · p_u − Y_trans,e · p_v
        # (manuscript convention — flow at u's end, includes wall-storage current)
        interior_nodes = prob['interior_nodes']
        boundary_nodes = prob['boundary_nodes']
        n_int = len(interior_nodes)
        interior_idx = {n: i for i, n in enumerate(interior_nodes)}
        boundary_idx = {n: i for i, n in enumerate(boundary_nodes)}
        edge_G = np.zeros(len(edges_in))
        edge_kappa_factor = np.zeros(len(edges_in), dtype=complex)
        # κ²_e / D_e = i · ω · 8μ · L² / R²       (stored once per edge)
        for k, (u, v) in enumerate(edges_in):
            if R_e[k] <= 0 or L_e[k] <= 0: continue
            edge_G[k] = float(np.pi * R_e[k] ** 4 / (8.0 * MU * L_e[k]))
            edge_kappa_factor[k] = (1j * 2.0 * np.pi * F0_HZ
                                     * 8.0 * MU
                                     * (L_e[k] ** 2) / (R_e[k] ** 2))
        # ω at the "primary" channel (drives the main T(D) build and the
        # κ²/C_e cap).  H₂-only mode runs at ω = 2·ω_H₁ ; joint mode uses
        # ω_H₁ as primary and adds a 2·ω_H₁ branch below.
        omega_h1 = 2.0 * np.pi * F0_HZ
        omega_h2 = 2.0 * omega_h1
        omega_primary = omega_h2 if primary_channel == 'H2' else omega_h1
        # κ_factor at the primary channel (used in the D-cap and in
        # _edge_admittances).  H₂ has 2× the storage current per edge.
        kappa_factor_primary = edge_kappa_factor * (
            2.0 if primary_channel == 'H2' else 1.0)

        def _edge_admittances(D_e_array, kappa_factor=None):
            """O(C_e) linearization of the distributed two-port — the
            polynomial form valid for C_e ≪ 1 (enforced by the D-grid
            cap above).  Accurate to O(C_e²) ~ 0.003% in our regime.

                Y_self,e  = G_e · (1 + κ²_e / 3)
                Y_trans,e = G_e · (1 − κ²_e / 6)

            with κ² = i·ω·D·8μ·L²/R² (= i·C_e) supplied per edge.
            Pass a different `kappa_factor` to evaluate at a different ω
            (e.g. 2·edge_kappa_factor for the H₂ channel)."""
            kf = kappa_factor if kappa_factor is not None else kappa_factor_primary
            kappa_sq = kf * D_e_array
            Y_self = edge_G * (1.0 + kappa_sq / 3.0)
            Y_trans = edge_G * (1.0 - kappa_sq / 6.0)
            # Force the resistive limit on degenerate / unused edges
            Y_self = np.where(edge_G > 0, Y_self, 0.0)
            Y_trans = np.where(edge_G > 0, Y_trans, 0.0)
            return Y_self, Y_trans

        def _T_H1_at_D(D0):
            """Build the distributed-two-port H1 admittance system at
            scalar D₀, with per-edge D_e = D₀·R_factor.  Returns the
            full (n_edges, n_B) complex transfer matrix where
              T[e, jb] = Y_self,e · p_u^{(jb)} − Y_trans,e · p_v^{(jb)}
            and p_u^{(jb)}, p_v^{(jb)} are the interior pressures driven
            by the jb-th unit boundary pressure."""
            D_e_arr = float(D0) * R_factor
            Y_self_arr, Y_trans_arr = _edge_admittances(D_e_arr)
            L_mat = np.zeros((n_int, n_int), dtype=complex)
            B_mat = np.zeros((n_int, n_B), dtype=complex)
            for k, (u, v) in enumerate(edges_in):
                if edge_G[k] == 0: continue
                Y_s = Y_self_arr[k]; Y_t = Y_trans_arr[k]
                u_int = u in interior_idx; v_int = v in interior_idx
                if u_int and v_int:
                    iu, iv = interior_idx[u], interior_idx[v]
                    L_mat[iu, iu] += Y_s; L_mat[iv, iv] += Y_s
                    L_mat[iu, iv] -= Y_t; L_mat[iv, iu] -= Y_t
                elif u_int and not v_int:
                    iu, jb = interior_idx[u], boundary_idx[v]
                    L_mat[iu, iu] += Y_s; B_mat[iu, jb] -= Y_t
                elif v_int and not u_int:
                    iv, jb = interior_idx[v], boundary_idx[u]
                    L_mat[iv, iv] += Y_s; B_mat[iv, jb] -= Y_t
            # Solve interior Laplacian; lstsq for rank-deficient safety
            diag_med = np.median(np.abs(np.diag(L_mat))) if n_int else 1.0
            scale = 1.0 / diag_med if diag_med > 0 else 1.0
            P_int = np.linalg.lstsq(L_mat * scale, -B_mat * scale,
                                     rcond=None)[0]
            # Edge flow readout: Q_e = Y_self·p_u − Y_trans·p_v
            T = np.zeros((len(edges_in), n_B), dtype=complex)
            for k, (u, v) in enumerate(edges_in):
                if edge_G[k] == 0: continue
                Y_s = Y_self_arr[k]; Y_t = Y_trans_arr[k]
                for jb in range(n_B):
                    pu = (P_int[interior_idx[u], jb] if u in interior_idx
                          else (1.0 if boundary_idx[u] == jb else 0.0))
                    pv = (P_int[interior_idx[v], jb] if v in interior_idx
                          else (1.0 if boundary_idx[v] == jb else 0.0))
                    T[k, jb] = Y_s * pu - Y_t * pv
            return T

        # ── Grid evaluation:  σ²_j = a² + s²·|Q_j|²  ──
        # Joint (a, s²) marginalisation on each D grid point.
        # In H₁+H₂ joint mode, the H₂ likelihood (with its own b̂_H₂
        # marginalised under the same Gaussian prior) is ADDED to the
        # H₁ log-likelihood at each (D, a, s²) grid point.  This is
        # valid because b̂_H₁ and b̂_H₂ are independent in the prior.
        #
        # ── Optional SNR floor (use_snr_floor=True) ──
        # The empirical (a, s²) fit can absorb model misfit as "noise"
        # and confidently down-weight edges that are actually fine.
        # The SNR floor prevents that: σ²_j ≥ |Q_j|² · 10^(-SNR_j/10),
        # which is the variance implied by the stored per-edge SNR.
        # We apply σ²_j = max(σ²_empirical, σ²_SNR) inside the loop.
        # Edges with no usable SNR get σ²_SNR = 0 → floor inactive.
        if use_snr_floor:
            snr_use_H1 = snr_per_edge[valid]
            sigma2_snr_H1 = np.where(np.isfinite(snr_use_H1),
                                       Q_abs2 * 10.0 ** (-snr_use_H1 / 10.0),
                                       0.0)
            if harmonic_mode == 'H1+H2' and Q_abs2_H2 is not None:
                snr_use_H2 = snr_H2_per_edge[valid]
                sigma2_snr_H2 = np.where(np.isfinite(snr_use_H2),
                                           Q_abs2_H2 * 10.0 ** (-snr_use_H2 / 10.0),
                                           0.0)
            elif primary_channel == 'H2':
                snr_use_H2 = snr_H2_per_edge[valid]
                sigma2_snr_H1 = np.where(np.isfinite(snr_use_H2),
                                           Q_abs2 * 10.0 ** (-snr_use_H2 / 10.0),
                                           0.0)
                sigma2_snr_H2 = None
            else:
                sigma2_snr_H2 = None
        else:
            sigma2_snr_H1 = None
            sigma2_snr_H2 = None

        kappa_factor_H2 = edge_kappa_factor * 2.0
        log_L_H1 = np.full((D_GRID.size, A_GRID.size, S2_GRID.size), -np.inf)
        log_L_H2 = np.full((D_GRID.size, A_GRID.size, S2_GRID.size), -np.inf)
        T_cache = {}
        T_H2_cache = {}
        for i, D0 in enumerate(D_GRID):
            try:
                T_full = _T_H1_at_D(float(D0))
                T_cache[i] = T_full
                T_sel = T_full[valid]
                n_r = T_sel.shape[0]
                T_R = np.zeros((2 * n_r, 2 * n_B))
                T_R[:n_r, :n_B] = T_sel.real; T_R[:n_r, n_B:] = -T_sel.imag
                T_R[n_r:, :n_B] = T_sel.imag; T_R[n_r:, n_B:] = T_sel.real
                # H₂ branch (joint mode only): build a separate T_R_H2 by
                # temporarily swapping the κ factor inside the closure.
                T_R_H2 = None
                if harmonic_mode == 'H1+H2':
                    # Re-evaluate the same forward-model build but at 2ω.
                    D_e_arr_H2 = float(D0) * R_factor
                    ks_H2 = kappa_factor_H2 * D_e_arr_H2
                    Y_self_H2 = edge_G * (1.0 + ks_H2 / 3.0)
                    Y_trans_H2 = edge_G * (1.0 - ks_H2 / 6.0)
                    Y_self_H2 = np.where(edge_G > 0, Y_self_H2, 0.0)
                    Y_trans_H2 = np.where(edge_G > 0, Y_trans_H2, 0.0)
                    L_mat_H2 = np.zeros((n_int, n_int), dtype=complex)
                    B_mat_H2 = np.zeros((n_int, n_B), dtype=complex)
                    for k, (u_, v_) in enumerate(edges_in):
                        if edge_G[k] == 0: continue
                        Y_s = Y_self_H2[k]; Y_t = Y_trans_H2[k]
                        u_int = u_ in interior_idx; v_int = v_ in interior_idx
                        if u_int and v_int:
                            iu, iv = interior_idx[u_], interior_idx[v_]
                            L_mat_H2[iu, iu] += Y_s; L_mat_H2[iv, iv] += Y_s
                            L_mat_H2[iu, iv] -= Y_t; L_mat_H2[iv, iu] -= Y_t
                        elif u_int and not v_int:
                            iu, jb = interior_idx[u_], boundary_idx[v_]
                            L_mat_H2[iu, iu] += Y_s; B_mat_H2[iu, jb] -= Y_t
                        elif v_int and not u_int:
                            iv, jb = interior_idx[v_], boundary_idx[u_]
                            L_mat_H2[iv, iv] += Y_s; B_mat_H2[iv, jb] -= Y_t
                    diag_med = (np.median(np.abs(np.diag(L_mat_H2)))
                                  if n_int else 1.0)
                    scale = 1.0 / diag_med if diag_med > 0 else 1.0
                    P_int_H2 = np.linalg.lstsq(L_mat_H2 * scale,
                                                 -B_mat_H2 * scale, rcond=None)[0]
                    T_H2 = np.zeros((len(edges_in), n_B), dtype=complex)
                    for k, (u_, v_) in enumerate(edges_in):
                        if edge_G[k] == 0: continue
                        Y_s = Y_self_H2[k]; Y_t = Y_trans_H2[k]
                        for jb in range(n_B):
                            pu = (P_int_H2[interior_idx[u_], jb] if u_ in interior_idx
                                  else (1.0 if boundary_idx[u_] == jb else 0.0))
                            pv = (P_int_H2[interior_idx[v_], jb] if v_ in interior_idx
                                  else (1.0 if boundary_idx[v_] == jb else 0.0))
                            T_H2[k, jb] = Y_s * pu - Y_t * pv
                    T_H2_cache[i] = T_H2
                    T_H2_sel = T_H2[valid]
                    T_R_H2 = np.zeros((2 * n_r, 2 * n_B))
                    T_R_H2[:n_r, :n_B] = T_H2_sel.real
                    T_R_H2[:n_r, n_B:] = -T_H2_sel.imag
                    T_R_H2[n_r:, :n_B] = T_H2_sel.imag
                    T_R_H2[n_r:, n_B:] = T_H2_sel.real
                # Compute the H₁ likelihood (always needed) and the H₂
                # likelihood (only in joint mode) on the SAME (a, s²) grid.
                # The two channels are INDEPENDENT in (a, s²) — each
                # gets its own per-channel noise model.  The factorisation
                # over independent priors lets us logsumexp each channel's
                # likelihood separately and then add the two log evidences.
                for ja, a_ in enumerate(A_GRID):
                    a2 = float(a_) ** 2
                    for js, s2 in enumerate(S2_GRID):
                        # H₁ branch.  With SNR-noise mode on, replace
                        # the empirical (â, ŝ) variance entirely with
                        # the SNR-implied σ²_j = |Q_j|²·10^(−SNR_j/10).
                        # The (a, s²) loop still runs but log L now
                        # doesn't depend on (a, s²) — they become
                        # marginalised over a constant likelihood, so
                        # they fall out of the posterior on D.
                        if sigma2_snr_H1 is not None:
                            sigma2 = sigma2_snr_H1.copy()
                            # Replace any zero / non-finite σ² (edges
                            # with missing SNR) with the empirical
                            # value so they still contribute.
                            bad = (sigma2 <= 0) | ~np.isfinite(sigma2)
                            if bad.any():
                                sigma2_emp = a2 + float(s2) * Q_abs2
                                sigma2[bad] = sigma2_emp[bad]
                        else:
                            sigma2 = a2 + float(s2) * Q_abs2
                        Sigma_diag = np.concatenate(
                            [0.5 * sigma2, 0.5 * sigma2])
                        C = np.diag(Sigma_diag) + T_R @ P0 @ T_R.T
                        C += 1e-10 * np.mean(np.diag(C)) * np.eye(C.shape[0])
                        try:
                            L = np.linalg.cholesky(0.5 * (C + C.T))
                            z = np.linalg.solve(L, y_H1)
                            log_L_H1[i, ja, js] = (-np.sum(np.log(np.diag(L)))
                                                    - 0.5 * (z @ z))
                        except np.linalg.LinAlgError:
                            pass
                        # H₂ branch — only in joint mode.  Independent
                        # noise parameters: each channel sees the same
                        # (a, s²) grid here, but they MARGINALISE
                        # independently below.
                        if harmonic_mode == 'H1+H2' and T_R_H2 is not None:
                            if sigma2_snr_H2 is not None:
                                sigma2_H2 = sigma2_snr_H2.copy()
                                bad = (sigma2_H2 <= 0) | ~np.isfinite(sigma2_H2)
                                if bad.any():
                                    s2e = a2 + float(s2) * Q_abs2_H2
                                    sigma2_H2[bad] = s2e[bad]
                            else:
                                sigma2_H2 = a2 + float(s2) * Q_abs2_H2
                            Sigma_diag_H2 = np.concatenate(
                                [0.5 * sigma2_H2, 0.5 * sigma2_H2])
                            C_H2 = np.diag(Sigma_diag_H2) + T_R_H2 @ P0 @ T_R_H2.T
                            C_H2 += 1e-10 * np.mean(np.diag(C_H2)) * np.eye(C_H2.shape[0])
                            try:
                                L_H2_ch = np.linalg.cholesky(0.5 * (C_H2 + C_H2.T))
                                z_H2 = np.linalg.solve(L_H2_ch, y_H2)
                                log_L_H2[i, ja, js] = (-np.sum(np.log(np.diag(L_H2_ch)))
                                                       - 0.5 * (z_H2 @ z_H2))
                            except np.linalg.LinAlgError:
                                pass
            except Exception:
                pass
        if not np.isfinite(log_L_H1).any():
            return {'error': 'Likelihood grid is all -inf — inference failed'}
        # Independent (a, s²) marginalisation per channel.  log p(D) is
        # the sum of per-channel log marginal likelihoods plus the D prior.
        log_p_D = logsumexp(log_L_H1, axis=(1, 2))
        if harmonic_mode == 'H1+H2' and np.isfinite(log_L_H2).any():
            log_p_D = log_p_D + logsumexp(log_L_H2, axis=(1, 2))
        log_p_D = log_p_D + log_prior_D
        log_p_D = log_p_D - log_p_D.max()  # normalise for stability
        # `log_L` is used by downstream MAP-(a, s²) code; in joint mode
        # we summarise as H1 + H2 so the heatmap still corresponds to the
        # full inference, but per-channel marginalisation has already
        # happened in log_p_D above.
        if harmonic_mode == 'H1+H2':
            log_L = log_L_H1 + log_L_H2
        else:
            log_L = log_L_H1
        j_map = int(np.argmax(log_p_D))
        D_map = float(D_GRID[j_map])
        # ── Two interval reports ──
        # (a) 3-nat posterior SUPPORT region: log p ≥ log p_max − 3.
        #     NOT a credible interval — its enclosed posterior mass is
        #     not 0.95 in general (Desktop critique).  Kept as a visual
        #     diagnostic because it cleanly separates SHARP / MOD / LOOSE
        #     tiers across the existing sweep results.
        band = log_p_D >= log_p_D[j_map] - 3.0
        lo = float(D_GRID[band][0]) if band.any() else float('nan')
        hi = float(D_GRID[band][-1]) if band.any() else float('nan')
        width = math.log10(hi / lo) if lo and lo > 0 else float('inf')
        # (b) Proper 90% (and 95%) credible intervals from the actual
        #     posterior mass on the log-D grid.  log_p_D values are
        #     evaluated as log p(x_D | y) on a uniformly-spaced log grid,
        #     so the bin widths are constant and posterior mass per bin
        #     is proportional to exp(log_p_D − max).
        post = np.exp(log_p_D - log_p_D.max())
        post = post / post.sum()
        # Highest-Posterior-Density intervals: sort bins by descending
        # mass, accumulate until the target probability is met, then
        # report the lo/hi grid endpoints of the included set.  For a
        # unimodal posterior this matches the standard HPD construction.
        def _hpd(p_target):
            order = np.argsort(-post)          # descending
            cum = np.cumsum(post[order])
            n_inc = int(np.searchsorted(cum, p_target) + 1)
            n_inc = min(n_inc, post.size)
            keep = np.sort(order[:n_inc])
            return float(D_GRID[keep[0]]), float(D_GRID[keep[-1]])
        lo_90, hi_90 = _hpd(0.90)
        lo_95, hi_95 = _hpd(0.95)
        width_90 = math.log10(hi_90 / lo_90) if lo_90 > 0 else float('inf')
        width_95 = math.log10(hi_95 / lo_95) if lo_95 > 0 else float('inf')
        interior = (j_map > 0 and j_map < D_GRID.size - 1)
        tier = ('SHARP' if (interior and width <= 0.5)
                else 'MODERATE' if (interior and width <= 1.0)
                else 'LOOSE' if (interior and width <= 2.0)
                else 'UN-LOC')
        # ── MAP (a, s²) per channel at the MAP D ──
        # The H₁ MAP is from log_L_H1, the H₂ MAP from log_L_H2 (joint
        # mode only).  Each channel has its OWN measurement-noise floor
        # â and proportional misfit ŝ — they don't share parameters.
        LL_H1_atD = log_L_H1[j_map]
        j_a_H1, j_s_H1 = np.unravel_index(int(np.argmax(LL_H1_atD)),
                                            LL_H1_atD.shape)
        a_map_H1 = float(A_GRID[j_a_H1])
        s2_map_H1 = float(S2_GRID[j_s_H1])
        s_map_H1 = float(np.sqrt(s2_map_H1))
        Q_crossover_H1 = a_map_H1 / max(s_map_H1, 1e-30)
        # Keep the legacy fields wired to the H₁ MAP for back-compat with
        # the diagnostic popup, compare popup, and downstream consumers.
        a_map = a_map_H1
        s2_map = s2_map_H1
        s_map = s_map_H1
        Q_crossover = Q_crossover_H1
        # H₂ MAP (joint mode)
        if harmonic_mode == 'H1+H2' and np.isfinite(log_L_H2).any():
            LL_H2_atD = log_L_H2[j_map]
            j_a_H2, j_s_H2 = np.unravel_index(int(np.argmax(LL_H2_atD)),
                                                LL_H2_atD.shape)
            a_map_H2 = float(A_GRID[j_a_H2])
            s2_map_H2 = float(S2_GRID[j_s_H2])
            s_map_H2 = float(np.sqrt(s2_map_H2))
            Q_crossover_H2 = a_map_H2 / max(s_map_H2, 1e-30)
        else:
            a_map_H2 = float('nan'); s2_map_H2 = float('nan')
            s_map_H2 = float('nan'); Q_crossover_H2 = float('nan')
        # ── MAP b̂_H₁ at (D_map, a_map_H1, s2_map_H1) ──
        T_at_map = T_cache[j_map]
        T_sel = T_at_map[valid]
        n_r = T_sel.shape[0]
        T_R = np.zeros((2 * n_r, 2 * n_B))
        T_R[:n_r, :n_B] = T_sel.real; T_R[:n_r, n_B:] = -T_sel.imag
        T_R[n_r:, :n_B] = T_sel.imag; T_R[n_r:, n_B:] = T_sel.real
        sigma2 = a_map_H1**2 + s2_map_H1 * Q_abs2
        Sigma_diag = np.concatenate([0.5 * sigma2, 0.5 * sigma2])
        try:
            A = T_R.T @ np.diag(1.0 / Sigma_diag) @ T_R + np.linalg.inv(P0)
            b_hat = np.linalg.solve(A, T_R.T @ (y_H1 / Sigma_diag))
        except np.linalg.LinAlgError:
            b_hat = np.zeros(2 * n_B)
        b_complex = b_hat[:n_B] + 1j * b_hat[n_B:]
        # Predicted H1 phasor at every edge in the carve (including bridges)
        Q_H1_pred = T_at_map @ b_complex
        # ── MAP b̂_H₂ at (D_map, a_map_H2, s2_map_H2) — joint mode only ──
        b_complex_H2 = None
        Q_H2_pred = None
        if harmonic_mode == 'H1+H2' and j_map in T_H2_cache:
            T_at_map_H2 = T_H2_cache[j_map]
            T_sel_H2 = T_at_map_H2[valid]
            T_R_H2 = np.zeros((2 * n_r, 2 * n_B))
            T_R_H2[:n_r, :n_B] = T_sel_H2.real
            T_R_H2[:n_r, n_B:] = -T_sel_H2.imag
            T_R_H2[n_r:, :n_B] = T_sel_H2.imag
            T_R_H2[n_r:, n_B:] = T_sel_H2.real
            sigma2_H2 = a_map_H2**2 + s2_map_H2 * Q_abs2_H2
            Sigma_diag_H2 = np.concatenate([0.5 * sigma2_H2, 0.5 * sigma2_H2])
            try:
                A_H2 = (T_R_H2.T @ np.diag(1.0 / Sigma_diag_H2) @ T_R_H2
                        + np.linalg.inv(P0))
                b_hat_H2 = np.linalg.solve(
                    A_H2, T_R_H2.T @ (y_H2 / Sigma_diag_H2))
                b_complex_H2 = b_hat_H2[:n_B] + 1j * b_hat_H2[n_B:]
                Q_H2_pred = T_at_map_H2 @ b_complex_H2
            except np.linalg.LinAlgError:
                b_complex_H2 = np.zeros(n_B, dtype=complex)
                Q_H2_pred = np.zeros(len(edges_in), dtype=complex)
        # ── Per-edge Fisher information about D₀ ──
        # I_j(D₀) = |∂Q_pred,j / ∂D₀|² / σ²_total,j        (at MAP b̂, MAP s²)
        # Numerically: central finite difference on T(D) and apply b̂.
        # Edges with high I_j drive D̂; edges with low I_j are passengers.
        # The sum across edges (Σ I_j) is the conditional Fisher info on
        # D₀ given b̂; CI width scales as 1/√Σ I_j.
        try:
            eps_D = 0.05 * D_map
            T_plus = _T_H1_at_D(D_map + eps_D)
            T_minus = _T_H1_at_D(max(D_map - eps_D, D_GRID[0]))
            dT_dD = (T_plus - T_minus) / (2.0 * eps_D)
            dQ_dD = dT_dD @ b_complex                      # complex, per-edge
            # σ²_total at MAP s² for the measured edges only.  For unmeasured
            # edges in the carve we report Fisher info = 0 (they don't appear
            # in y, so they have no Fisher contribution about D₀).
            fisher_per_edge = np.zeros(len(edges_in))
            for k_e in range(len(edges_in)):
                if not (prob['valid_dc'][k_e] and prob['valid_h1'][k_e]):
                    continue
                Q_obs_k = prob['Q_H1_obs'][k_e]
                Q_abs2_k = float(np.abs(Q_obs_k) ** 2)
                # σ²_total at MAP = a² + s²·|Q|²  (matches the marginal LL)
                sigma2_k = a_map**2 + s2_map * Q_abs2_k
                if sigma2_k <= 0: continue
                fisher_per_edge[k_e] = float(
                    (dQ_dD[k_e].real ** 2 + dQ_dD[k_e].imag ** 2)
                    / sigma2_k)
        except Exception:
            fisher_per_edge = np.zeros(len(edges_in))
        # ── DC pressure solve at boundary nodes ──
        # At DC (ω = 0) the distributed two-port reduces exactly to the
        # purely resistive limit:  κ → 0,  Y_self → G,  Y_trans → G.
        # Build the DC admittance from scratch using the same
        # distributed framework with D₀ = 0 → κ = 0 → resistive limit.
        # This guarantees the DC and H₁ solves use the SAME forward map
        # (manuscript form), not a separate lumped variant.
        b_DC = np.zeros(n_B)
        # Diagnostic-only DC residuals (kept None if the DC solve below
        # is skipped or fails — the H₁ inference is unaffected either way).
        Q_DC_meas_diag = None
        Q_DC_pred_diag = None
        Q_DC_pred_full = None
        sigma_DC_diag = None
        valid_dc_idx_diag = None
        try:
            L_DC = np.zeros((n_int, n_int), dtype=float)
            B_DC_mat = np.zeros((n_int, n_B), dtype=float)
            for k, (uu, vv) in enumerate(edges_in):
                if edge_G[k] == 0: continue
                G = edge_G[k]      # κ → 0 limit: Y_self = Y_trans = G
                u_int = uu in interior_idx; v_int = vv in interior_idx
                if u_int and v_int:
                    iu, iv = interior_idx[uu], interior_idx[vv]
                    L_DC[iu, iu] += G; L_DC[iv, iv] += G
                    L_DC[iu, iv] -= G; L_DC[iv, iu] -= G
                elif u_int and not v_int:
                    iu, jb = interior_idx[uu], boundary_idx[vv]
                    L_DC[iu, iu] += G; B_DC_mat[iu, jb] -= G
                elif v_int and not u_int:
                    iv, jb = interior_idx[vv], boundary_idx[uu]
                    L_DC[iv, iv] += G; B_DC_mat[iv, jb] -= G
            diag_med_DC = (np.median(np.abs(np.diag(L_DC)))
                            if n_int else 1.0)
            scale_DC = 1.0 / diag_med_DC if diag_med_DC > 0 else 1.0
            P_int_DC = np.linalg.lstsq(
                L_DC * scale_DC, -B_DC_mat * scale_DC, rcond=None)[0]
            T_DC = np.zeros((len(edges_in), n_B))
            for k, (uu, vv) in enumerate(edges_in):
                G = edge_G[k]
                if G == 0: continue
                for jb in range(n_B):
                    pu = (P_int_DC[interior_idx[uu], jb] if uu in interior_idx
                          else (1.0 if boundary_idx[uu] == jb else 0.0))
                    pv = (P_int_DC[interior_idx[vv], jb] if vv in interior_idx
                          else (1.0 if boundary_idx[vv] == jb else 0.0))
                    # DC limit of Q_e = Y_self·p_u − Y_trans·p_v → G·(p_u − p_v)
                    T_DC[k, jb] = G * (pu - pv)
            valid_dc_idx = np.where(prob['valid_dc'])[0]
            if valid_dc_idx.size >= n_B:
                T_DC_sel = T_DC[valid_dc_idx]
                Q_DC_meas = prob['Q_DC_obs'][valid_dc_idx]
                # Per-edge DC measurement variance from canonical SNR_DC
                # (same convention as H₁):  σ²_DC,j = |Q_DC,j|²·10^(−SNR_DC/10).
                # Fall back to a 30% relative floor for edges where SNR_DC
                # isn't populated.  This is just the LS weighting for the
                # post-hoc b̂_DC solve — it doesn't enter the H₁ inference.
                snr_DC_sel = snr_DC_per_edge[valid_dc_idx]
                snr_DC_sel = np.where(np.isfinite(snr_DC_sel),
                                        snr_DC_sel, np.nan)
                r_DC_sel = 10.0 ** (-snr_DC_sel / 10.0)
                sigma_DC_emp = np.abs(Q_DC_meas) * np.sqrt(r_DC_sel)
                # Where SNR is missing → fall back to 30 % relative
                fallback = 0.3 * np.abs(Q_DC_meas)
                sigma_DC = np.where(np.isfinite(sigma_DC_emp),
                                      sigma_DC_emp, fallback)
                # Hard floor against zero-magnitude edges
                sigma_DC = np.maximum(sigma_DC, 1e-18)
                W_DC = 1.0 / sigma_DC ** 2
                ATA = T_DC_sel.T @ (W_DC[:, None] * T_DC_sel)
                ATb = T_DC_sel.T @ (W_DC * Q_DC_meas)
                diag_scale = (np.median(np.abs(np.diag(ATA)))
                                if n_B else 1.0) or 1.0
                ATA += 1e-3 * diag_scale * np.eye(n_B)
                b_DC = np.linalg.solve(ATA, ATb)
                # DC gauge fix: the absolute pressure at boundary nodes
                # is defined only up to a constant.  The ridge term
                # selected an arbitrary minimum-norm gauge.  For
                # visualisation we subtract the mean so that displayed
                # |b̂_DC,n| reflects spatial pressure VARIATION rather
                # than the gauge offset.  Differences between BC nodes
                # are gauge-invariant; absolute values are not.
                b_DC = b_DC - float(np.mean(b_DC))
                # Stash DC residual ingredients for the diagnostic panel.
                # Q_DC prediction is gauge-invariant (T_DC rows are p_u−p_v
                # differences), so the mean-subtraction above doesn't matter.
                Q_DC_pred_diag = T_DC_sel @ b_DC
                Q_DC_meas_diag = Q_DC_meas
                sigma_DC_diag = sigma_DC
                valid_dc_idx_diag = valid_dc_idx
                # Also store full-edges-in Q_DC prediction (parallel
                # to Q_H1_pred) so the Compare popup can show the
                # *inferred* DC mean alongside the measured one.
                Q_DC_pred_full = T_DC @ b_DC
        except Exception:
            pass
        delta_at_zero = float(
            logsumexp(log_L[j_map]) - logsumexp(log_L[0]))
        return dict(
            tile_id=int(tile_id),
            alpha=float(alpha),
            sigma_b=float(sigma_b_Pa),
            bc_length_px=float(bc_length_px),
            n_av_promoted=int(_n_av_promoted),
            # Physical-validity D cap based on max_e(C_e) ≤ C_E_CAP
            C_e_cap=float(C_E_CAP),
            D_cap=float(D_cap),
            cap_omega_mult=(None if cap_omega_mult is None
                              else float(cap_omega_mult)),
            K_meas_max=float(K_meas_max),
            # C_e at MAP — sanity check on whether the chosen D is well
            # inside the linear-RC regime.
            C_e_at_map=float(D_map * K_meas_max),
            D_hat=D_map,
            lo=lo, hi=hi, width=width,             # 3-nat support region
            lo_90=lo_90, hi_90=hi_90, width_90=width_90,   # HPD 90% CI
            lo_95=lo_95, hi_95=hi_95, width_95=width_95,   # HPD 95% CI
            tier=tier,
            s2_at_map=s2_map,
            a_at_map=a_map,
            s_at_map=s_map,
            Q_crossover=Q_crossover,
            frac_err_pct=float(np.sqrt(s2_map) * 100),
            log_p_D=log_p_D,
            D_GRID=D_GRID,
            log_L=log_L,
            S2_GRID=S2_GRID,
            A_GRID=A_GRID,
            n_meas=n_v,
            snr_median=float(np.nanmedian(snr_per_edge[valid_idx])),
            R_ref_um=float(R_ref * 1e6),
            R_min_um=float(R_e[valid].min() * 1e6),
            R_max_um=float(R_e[valid].max() * 1e6),
            b_hat=b_hat,
            b_complex=b_complex,
            b_DC=b_DC,
            # DC residual ingredients for the diagnostic (None if DC solve
            # was skipped — Panel D will just plot H₁ in that case).
            Q_DC_meas_diag=Q_DC_meas_diag,
            Q_DC_pred_diag=Q_DC_pred_diag,
            Q_DC_pred_full=Q_DC_pred_full,
            harmonic_mode=str(harmonic_mode),
            # Per-channel noise-model MAP values (joint-mode only for H2)
            a_at_map_H1=a_map_H1,
            s_at_map_H1=s_map_H1,
            Q_crossover_H1=Q_crossover_H1,
            a_at_map_H2=a_map_H2,
            s_at_map_H2=s_map_H2,
            Q_crossover_H2=Q_crossover_H2,
            # H₂ MAP forcing + prediction (joint mode only)
            b_complex_H2=b_complex_H2,
            Q_H2_pred=Q_H2_pred,
            sigma_DC_diag=sigma_DC_diag,
            valid_dc_idx_diag=valid_dc_idx_diag,
            Q_H1_pred=Q_H1_pred,
            # Fisher info is stored in BOTH units:
            #   fisher_per_edge       : I_j(D)        in 1/Pa²   (raw)
            #   fisher_per_edge_logD  : I_j(log D)    dimensionless,
            #     converted via chain rule  ∂Q/∂(log D) = D · ∂Q/∂D
            #     so I_log_j = I_D_j · D̂²
            #     Directly comparable to the log-normal prior precision
            #     1/τ_D² = 1/(ln 10)² ≈ 0.189.
            fisher_per_edge=fisher_per_edge,
            fisher_per_edge_logD=fisher_per_edge * (D_map ** 2),
            R_e=R_e,
            prob=prob,
            delta_at_zero=delta_at_zero,
        )


    def _inference_render_diagnostic(self, tile_id, result):
        """Six-panel diagnostic for the additive-noise Bayesian fit:
          (1) D₀ posterior with the log-normal prior overlaid,
          (2) priors + physical assumptions as a text panel,
          (3) 2D log-likelihood heatmap over (D₀, s²),
          (4) normalised residual histogram (H₁ Re∪Im + DC overlay)
              against N(0,1) reference,
          (5) per-edge fit scatter Re(H₁),
          (6) per-edge fit scatter Im(H₁).
        """
        import matplotlib.pyplot as plt
        import tempfile
        D_GRID = result['D_GRID']
        log_p_D = result['log_p_D']
        log_L = result['log_L']
        S2_GRID = result['S2_GRID']
        D_hat = result['D_hat']
        lo, hi = result['lo'], result['hi']
        s2_map = result['s2_at_map']
        frac_pct = result['frac_err_pct']
        prob = result['prob']
        b_complex = result['b_complex']
        # Physical constants (mirror values used in _inference_run)
        MU_PHYS = 3.5e-3
        RHO_PHYS = 1.06e3
        PX_SIZE_M_PHYS = 1.7e-6
        tile_f0_map = self.G.graph.get('tile_f0_piv') or {}
        f0_used = float(tile_f0_map.get(int(tile_id))
                         or self.G.graph.get('f0_hz') or 2.5)
        R_ref_m = result['R_ref_um'] * 1e-6
        # Womersley at R_ref: Wo = R√(ωρ/μ)
        omega = 2.0 * np.pi * f0_used
        Wo_ref = R_ref_m * np.sqrt(omega * RHO_PHYS / MU_PHYS)
        # RC attenuation length (Desktop-corrected).  For a distributed RC
        # vessel with r' = 8μ/(πR⁴), c' = πR²D, the propagation constant
        # γ = √(iωr'c') = √(i·8μωD/R²).  Re(γ) = √(4μωD/R²), so
        # the amplitude attenuation / phase accumulation length is
        #   ℓ_RC = 1/Re(γ) = R / (2·√(μ·ω·D)).
        # MORE compliance → SHORTER ℓ_RC (more storage, stronger damping).
        # Earlier display formula had D in the numerator; that was a
        # dimensional / physical error in the label only, not in the
        # inference itself.
        ell_ref_m = R_ref_m / max(
            2.0 * np.sqrt(MU_PHYS * omega * D_hat), 1e-30)

        fig = plt.figure(figsize=(11, 9), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])

        # ── Panel A: D₀ posterior with prior ──
        ax = fig.add_subplot(gs[0, 0])
        p_D = np.exp(log_p_D - log_p_D.max())
        MU_D, TAU_D = np.log(1.5e-3), np.log(10.0)
        log_prior_D = -0.5 * ((np.log(D_GRID) - MU_D) / TAU_D) ** 2
        prior = np.exp(log_prior_D - log_prior_D.max())
        ax.plot(D_GRID, prior, color='C7', lw=1.4, ls=':',
                 label=('prior: LogN(ln 1.5×10⁻³, τ=ln 10)'))
        # 95% prior support band
        D_prior_lo = np.exp(MU_D - 2 * TAU_D)
        D_prior_hi = np.exp(MU_D + 2 * TAU_D)
        ax.axvspan(D_prior_lo, D_prior_hi, color='C7', alpha=0.06,
                    label=f'95% prior band [{D_prior_lo:.1e}, {D_prior_hi:.1e}]')
        ax.fill_between(D_GRID, 0, p_D, alpha=0.25, color='C0')
        ax.plot(D_GRID, p_D, color='C0', lw=2.0,
                 label='posterior p(D₀ | y)')
        ax.axvline(D_hat, color='C3', lw=1.4, ls='--',
                    label=f'MAP D̂ = {D_hat:.2e}')
        if np.isfinite(lo) and np.isfinite(hi):
            ax.axvspan(lo, hi, color='C3', alpha=0.10,
                        label=f'3-nat support [{lo:.1e}, {hi:.1e}]')
        # Proper 90% HPD credible interval (Desktop critique fix: the
        # 3-nat band is not a probability-calibrated CI).
        lo_90 = result.get('lo_90'); hi_90 = result.get('hi_90')
        if lo_90 is not None and hi_90 is not None and np.isfinite(lo_90):
            ax.axvspan(lo_90, hi_90, color='C0', alpha=0.18,
                        label=f'90% HPD CI [{lo_90:.1e}, {hi_90:.1e}]')
        ax.set_xscale('log')
        ax.set_xlabel('D₀ (/Pa)')
        ax.set_ylabel('density (peak-normalised)')
        ax.set_title(f'D₀ posterior  ·  tile {tile_id}  ·  α = {result["alpha"]:+.2f}  ·  {result["tier"]}')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, which='both')

        # (Old Panel B text block dropped — clutter without much value.
        # Headline numbers are in the suptitle / panel titles instead.)
        ax_info = None  # unused, prior code referenced this name
        _info_lines_unused = [
            '── Forward model — distributed two-port (manuscript form) ──',
            '   Y_self,e  = G_e · (1 + κ_e²/3)',
            '   Y_trans,e = G_e · (1 − κ_e²/6)',
            '   κ_e² = i·ω·D_e·8μ L_e²/R_e²   ( = i·C_e ;  C_e = compliance number )',
            '   Edge flow:  Q_e = Y_self,e · p_u  −  Y_trans,e · p_v',
            f'   D-grid capped at  max_e C_e ≤ {result["C_e_cap"]:.2f}  '
            f'(D_cap = {result["D_cap"]:.2e}/Pa).',
            f'   At MAP D̂, max_e C_e ≈ {result["C_e_at_map"]:.3f}.',
            '',
            '── Priors ──',
            '   D₀  ~  LogNormal( μ = ln 1.5×10⁻³,  τ = ln 10 )',
            f'        ≈95% prior support: [{D_prior_lo:.1e}, {D_prior_hi:.1e}] /Pa',
            f'   b_H₁ ~  Complex N(0, σ_b² · I_{{n_B}})  with σ_b = {result["sigma_b"]:.1f} Pa',
            '        Var(Re b_n) = Var(Im b_n) = σ_b²/2;  σ_b = RMS complex amplitude.',
            f'   a   log-uniform on [{result["A_GRID"][0]:.0e}, {result["A_GRID"][-1]:.0e}]  '
            f'(m³/s, {result["A_GRID"].size} pts)',
            f'   s²  log-uniform on [{S2_GRID[0]:.0e}, {S2_GRID[-1]:.0e}]  '
            f'(dimensionless, {S2_GRID.size} pts)',
            '',
            '── Noise model (additive + proportional, both inferred) ──',
            '   σ²_j  =  a²  +  s² · |Q_H₁,j|²',
            '   a   (m³/s)        — measurement-noise FLOOR (absolute, signal-independent)',
            '   s²  (dimensionless) — proportional MISFIT (ridge + KCL deficit ∝ |Q|)',
            '   |Q|* = a / s       — crossover: |Q| < |Q|* → noise-dominated edge',
            '                                    |Q| ≥ |Q|* → misfit-dominated edge',
            f'   â = {result["a_at_map"]*1e12:.4f} nL/s   ·   '
            f'ŝ = {result["s_at_map"]:.3f}   ·   '
            f'|Q|* = {result["Q_crossover"]*1e12:.4f} nL/s',
            f'   median SNR_H₁ = {result["snr_median"]:+.1f} dB '
            f'(reported, NOT used in weighting; γ → 0 from inferred-shape test)',
            '',
            '── Identifiability note ──',
            '   Under C_e ≪ 1 with approximately-uniform b̂_H₁, the data identify',
            '   the product D·|b̂_H₁| and the shape of b̂ across BC nodes; the',
            '   absolute split between D and |b̂_H₁| depends on σ_b.',
            f'   D̂ above is the value at σ_b = {result["sigma_b"]:.1f} Pa.  Treat as one',
            '   point on a structural b–D ridge, not a unique measurement.',
            '',
            '── Per-edge D ──',
            f'   D_e = D₀ · (R_e / R_ref)^α   with α = {result["alpha"]:+.2f}',
            f'   R_ref (median) = {result["R_ref_um"]:.1f} µm  ·  '
            f'range [{result["R_min_um"]:.1f}, {result["R_max_um"]:.1f}] µm',
            '',
            '── Physical constants ──',
            f'   μ  (viscosity)  = {MU_PHYS*1e3:.1f} mPa·s',
            f'   ρ  (density)   = {RHO_PHYS:.0f} kg/m³',
            f'   px size        = {PX_SIZE_M_PHYS*1e6:.2f} µm/px',
            f'   f₀  (this tile) = {f0_used:.3f} Hz   (per-tile from canonical graph)',
            '',
            '── Derived ──',
            f'   ω = 2π·f₀          = {omega:.2f}  rad/s',
            f'   Wo at R_ref        = R·√(ωρ/μ)  = {Wo_ref:.3f}   '
            f'(quasi-steady Poiseuille valid if Wo small)',
            f'   RC propagation depth  ℓ_RC = R/(2·√(μωD))  '
            f'at R_ref = {ell_ref_m*1e6:.1f} µm',
            '',
            '── Carve ──',
            f'   {len(prob["edges_in"])} edges in carve  '
            f'({result["n_meas"]} measured H₁)',
            f'   {len(prob["boundary_nodes"])} boundary, '
            f'{len(prob["interior_nodes"])} interior nodes',
            (f'   ({result.get("n_av_promoted", 0)} anatomical A/V '
             f'anchors promoted from interior → boundary)'
             if result.get("n_av_promoted", 0) else
             '   (no anatomical A/V anchors in this carve)'),
        ]
        # (text panel removed)

        # ── Panel B: 2D (a, s²) log-likelihood at MAP D ──
        # Slice the 3D log_L at MAP D, plot the (a, s²) joint posterior.
        # Use an adaptive display range (tighter than the old hard-coded
        # -20) and overlay 1/2/3-σ contours so the actual constrained
        # region is visible even when the posterior is very peaked.
        ax = fig.add_subplot(gs[0, 1])
        A_GRID = result["A_GRID"]
        LL_atD = log_L[int(np.argmax(result["log_p_D"]))]
        log_L_disp = LL_atD - LL_atD.max()
        A_disp = A_GRID * 1e12          # nL/s
        # Adaptive vmin: cover the 99th-percentile of log p mass on grid
        # (so most of the colourmap range falls on the constrained region,
        # not the -∞ floor that dominated the old display).
        finite_ll = log_L_disp[np.isfinite(log_L_disp)]
        vmin = float(np.percentile(finite_ll, 1)) if finite_ll.size else -10.0
        vmin = max(vmin, -10.0)         # never go below -10 nats for clarity
        im = ax.pcolormesh(A_disp, S2_GRID, log_L_disp.T,
                            cmap='viridis', shading='auto',
                            vmin=vmin, vmax=0)
        plt.colorbar(im, ax=ax, label='log p(a, s² | y) − max  (nats)')
        # 1/2/3-σ contour boundaries: Δlog p ≈ -0.5 / -2 / -4.5 in 2D.
        try:
            ax.contour(A_disp, S2_GRID, log_L_disp.T,
                        levels=[-4.5, -2.0, -0.5],
                        colors='white', linewidths=[0.5, 0.8, 1.1],
                        linestyles=['dotted', 'dashed', 'solid'])
        except Exception:
            pass
        a_disp = result["a_at_map"] * 1e12
        ax.plot(a_disp, s2_map, 'o', color='red', mfc='none', ms=9,
                 mew=2, label=f'MAP  â = {a_disp:.3f} nL/s,  ŝ = {result["s_at_map"]:.2g}')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('a  (nL/s, measurement noise floor)')
        ax.set_ylabel('s²  (proportional misfit)')
        ax.set_title('(a, s²) joint posterior at MAP D')
        ax.legend(loc='upper right', fontsize=9)

        # ── Per-edge fit data + per-edge SNR (used in panels D, E, F) ──
        valid = prob['valid_dc'] & prob['valid_h1']
        Q_meas = prob['Q_H1_obs'][valid]
        Q_pred = result['Q_H1_pred'][valid]
        n_v = int(valid.sum())
        snrs = np.zeros(n_v)
        valid_idx = np.where(valid)[0]
        for k, i in enumerate(valid_idx):
            u, v = prob['edges_in'][i]
            for m in (self.G.edges[u, v].get('measurements_piv') or []):
                # v4 migration: was m.get('tile_id')
                if MeasurementView(m).tile_id == int(tile_id):
                    # Q_H1_snr_db is a canonical top-level field; keep.
                    snrs[k] = m.get('Q_H1_snr_db', 0.0) or 0.0; break

        # ── Panel C: Normalised residual histogram (H₁ + DC overlay) ──
        ax_hist = fig.add_subplot(gs[1, 0])
        Q_abs2 = np.abs(Q_meas) ** 2
        # σ²_total per edge = a² + s²·|Q|²;  per-component σ²/2
        a_map = float(result["a_at_map"])
        sigma_per_comp = np.sqrt(0.5 * (a_map**2 + s2_map * Q_abs2))
        sigma_per_comp = np.where(sigma_per_comp > 0, sigma_per_comp, 1e-30)
        z_re = (Q_meas.real - Q_pred.real) / sigma_per_comp
        z_im = (Q_meas.imag - Q_pred.imag) / sigma_per_comp
        z_H1 = np.concatenate([z_re, z_im])

        # DC residuals from the same Bayesian fit's b̂_DC solve
        # (post-hoc LS at the H₁ MAP).  σ_DC was built from canonical
        # SNR_DC per edge, same convention as H₁: σ²_DC,j = |Q_DC,j|²·10^(−SNR_DC/10).
        Q_DC_meas_d = result.get('Q_DC_meas_diag')
        Q_DC_pred_d = result.get('Q_DC_pred_diag')
        sigma_DC_d  = result.get('sigma_DC_diag')
        z_DC = None
        if (Q_DC_meas_d is not None and Q_DC_pred_d is not None
                and sigma_DC_d is not None and len(Q_DC_meas_d) > 0):
            sd = np.where(sigma_DC_d > 0, sigma_DC_d, 1e-30)
            z_DC = (Q_DC_meas_d - Q_DC_pred_d) / sd
            z_DC = np.asarray(z_DC, float)

        # Symmetric x limit covers whichever channel has fatter tails
        _z_pool = z_H1 if z_DC is None else np.concatenate([z_H1, z_DC])
        z_lim = max(4.0, np.percentile(np.abs(_z_pool), 95) * 1.2)
        bins = np.linspace(-z_lim, z_lim, 60)

        ax_hist.hist(z_H1, bins=bins, density=True,
                      histtype='stepfilled', color='C0', alpha=0.55,
                      edgecolor='C0', linewidth=1.3,
                      label=f'H₁ (Re ∪ Im, n = {z_H1.size})')
        if z_DC is not None:
            ax_hist.hist(z_DC, bins=bins, density=True,
                          histtype='stepfilled', color='C2', alpha=0.45,
                          edgecolor='C2', linewidth=1.3,
                          label=f'DC (n = {z_DC.size})')
        x_ref = np.linspace(-z_lim, z_lim, 200)
        gauss_ref = np.exp(-0.5 * x_ref ** 2) / np.sqrt(2 * np.pi)
        ax_hist.plot(x_ref, gauss_ref, color='C3', lw=1.6, ls='--',
                      label='N(0, 1)')
        ax_hist.axvline(0, color='#444', lw=0.4, ls=':')
        ax_hist.set_xlim(-z_lim, z_lim)
        ax_hist.set_xlabel('normalised residual  z = (Q_meas − Q_pred) / σ')
        ax_hist.set_ylabel('density')
        rms_H1 = float(np.sqrt(np.mean(z_H1 ** 2)))
        title_str = f'Residuals  ·  RMS|z|_H₁ = {rms_H1:.2f} (target 1)'
        if z_DC is not None and z_DC.size > 0:
            rms_DC = float(np.sqrt(np.mean(z_DC ** 2)))
            title_str += f'  ·  RMS|z|_DC = {rms_DC:.2f}'
        ax_hist.set_title(title_str, fontsize=11)
        ax_hist.legend(fontsize=9, loc='upper right')
        ax_hist.grid(True, alpha=0.25)

        # ── Panel D: per-edge fit Q_meas vs Q_pred (Re ∪ Im stacked) ──
        ax_fit = fig.add_subplot(gs[1, 1])
        scale = 1e12   # m³/s → nL/s
        meas_stack = np.concatenate([Q_meas.real, Q_meas.imag]) * scale
        pred_stack = np.concatenate([Q_pred.real, Q_pred.imag]) * scale
        snr_stack = np.concatenate([snrs, snrs])
        amp_max = float(max(np.max(np.abs(meas_stack)),
                             np.max(np.abs(pred_stack)), 0.01))
        sc = ax_fit.scatter(meas_stack, pred_stack,
                             c=snr_stack, cmap='viridis', s=14,
                             edgecolor='black', linewidth=0.25)
        plt.colorbar(sc, ax=ax_fit, label='SNR_H₁ (dB)')
        ax_fit.plot([-amp_max, amp_max], [-amp_max, amp_max],
                     'k--', lw=0.6, alpha=0.6)
        ax_fit.set_xlim(-amp_max, amp_max); ax_fit.set_ylim(-amp_max, amp_max)
        ax_fit.set_xlabel('measured  Q_H₁  (nL/s)')
        ax_fit.set_ylabel('predicted Q_H₁  (nL/s)')
        ax_fit.set_title('Per-edge fit: Re ∪ Im (H₁)')
        ax_fit.set_aspect('equal'); ax_fit.grid(True, alpha=0.25)

        fig.suptitle(
            f"Distributed-RC Bayesian D₀ inference  ·  tile {tile_id}  ·  "
            f"α = {result['alpha']:+.2f}  ·  σ_b = {result['sigma_b']:.1f} Pa  ·  "
            f"D̂ = {D_hat:.2e} /Pa  ·  3-nat width = {result['width']:.2f} dec  ·  "
            f"{result['tier']}",
            fontsize=11, fontweight='bold')

        # Save to PNG then display via Qt dialog (napari event loop)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as _tmp:
            _tmp_path = _tmp.name
        fig.savefig(_tmp_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        self._show_diagnostic_png(_tmp_path, int(tile_id), result)


    def _show_diagnostic_png(self, png_path: str, tile_id: int, result: dict):
        """Pop up a Qt dialog displaying a saved diagnostic PNG.

        The dialog is non-modal, has a fixed window title with the
        tile_id and headline D̂, and is retained on `self` so Python's
        GC doesn't close it the instant this method returns."""
        from qtpy.QtWidgets import (
            QDialog, QVBoxLayout, QLabel, QScrollArea)
        from qtpy.QtGui import QPixmap
        from qtpy.QtCore import Qt
        dlg = QDialog(self.viewer.window._qt_window)
        # Window title summarises the result; width-in-decades replaces
        # the old σ_D/D̂ relative-error display since the new Bayesian
        # posterior reports a 3-nat CI band, not a σ.
        width_str = (f"width {result['width']:.2f} dec"
                     if 'width' in result
                     else f"σ_D/D̂ = {result.get('rel_sigma_D', float('nan')):.1%}")
        dlg.setWindowTitle(
            f"Inference diagnostic — tile {tile_id}  "
            f"D̂ = {result['D_hat']:.3e},  {width_str}")
        # Layout: scroll area → label → pixmap
        lay = QVBoxLayout(dlg)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        lbl = QLabel()
        pixmap = QPixmap(png_path)
        lbl.setPixmap(pixmap)
        lbl.setAlignment(Qt.AlignCenter)
        scroll.setWidget(lbl)
        lay.addWidget(scroll)
        # Match the rendered figure size (default render dpi=180 →
        # 15×9 inches → ~2700×1620 px) but cap to 90% of screen.
        try:
            screen = self.viewer.window._qt_window.screen().availableGeometry()
            target_w = min(int(pixmap.width()),
                           int(0.9 * screen.width()))
            target_h = min(int(pixmap.height() + 50),
                           int(0.9 * screen.height()))
        except Exception:
            target_w, target_h = 1200, 800
        dlg.resize(target_w, target_h)
        # Keep a reference so the dialog stays alive after this method
        # returns.  Reuse a single list slot so old dialogs are GC'd
        # naturally if the user closes them.
        if not hasattr(self, '_inf_diag_dialogs'):
            self._inf_diag_dialogs = []
        # Prune any that have already been closed.
        self._inf_diag_dialogs = [
            d for d in self._inf_diag_dialogs if d.isVisible()]
        self._inf_diag_dialogs.append(dlg)
        dlg.show()


    # ── UI glue ──────────────────────────────────────────────────────
    def _on_inference_tile_change(self, _idx=None):
        # Stale: previous result was for a different tile.
        self._inf_diag_btn.setEnabled(False)
        if hasattr(self, '_inf_compare_btn'):
            self._inf_compare_btn.setEnabled(False)
        self._inf_results_box.setPlainText("")
        self._inference_last_result = None
        # Drop the BC forcing + Fisher-info overlays from previous tile.
        self._tear_down_inference_bc_overlay()
        self._tear_down_inference_fisher_overlay()
        if self._inf_show_carve_chk.isChecked():
            self._refresh_inference_carve_overlay()


    def _on_inference_show_carve_toggle(self, checked: bool):
        if not checked:
            self._tear_down_inference_carve_overlay()
        else:
            self._refresh_inference_carve_overlay()


    def _refresh_inference_carve_overlay(self):
        """Update the carve highlight on the mosaic — both the carved
        edges (thick yellow lines) and the carve boundary nodes (filled
        orange dots) — to reflect the currently-selected tile."""
        self._tear_down_inference_carve_overlay()
        tile_id = self._inf_tile_combo.currentData()
        if tile_id is None:
            return
        topo = self._inference_get_carve_topology(int(tile_id))
        edges = topo['edges']
        bnodes = topo['boundary_nodes']
        H = self.mosaic_height
        # ── Edge overlay (thick yellow lines) ──
        lines = []
        for (u, v) in edges:
            if not self.G.has_edge(u, v):
                continue
            try:
                x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
                x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            except KeyError:
                continue
            lines.append(np.array([[H - y1, x1], [H - y2, x2]]))
        if lines:
            self._inference_carve_layer = self.viewer.add_shapes(
                lines, shape_type='line',
                edge_color='#ffd400',  # bright yellow
                edge_width=8.0,
                opacity=0.85,
                name=f'Inference carve (tile {tile_id})',
            )
        # ── Boundary-node overlay (filled dots) ──
        pts = []
        for n in bnodes:
            try:
                x = self.G.nodes[n]['x']
                y = self.G.nodes[n]['y']
            except KeyError:
                continue
            pts.append([H - y, x])
        if pts:
            self._inference_carve_node_layer = self.viewer.add_points(
                np.asarray(pts), size=18,
                face_color='#ff6600',  # orange
                border_color='white',
                opacity=0.95,
                name=f'Carve boundary nodes (tile {tile_id})',
            )


    def _tear_down_inference_carve_overlay(self):
        for attr in ('_inference_carve_layer',
                     '_inference_carve_node_layer'):
            lyr = getattr(self, attr, None)
            if lyr is None:
                continue
            try:
                self.viewer.layers.remove(lyr)
            except (KeyError, ValueError):
                pass
            setattr(self, attr, None)


    def _refresh_inference_bc_overlay(self, result):
        """After an inference run, draw the inferred boundary forcing
        on the mosaic as two napari layers:

          - Points layer at each BC node, with
              face_color = |b̂_DC|   (DC pressure magnitude, viridis)
              size      = |b̂_H₁|    (H₁ pressure amplitude)
          - Vectors layer of arrows, one per BC node:
              origin    = BC node
              direction = arg(b̂_H₁)  (H₁ phase, math convention)
              length    ∝ |b̂_H₁|     so amplitude redundant-encoded

        Removes any previous BC layers first."""
        from matplotlib import cm
        self._tear_down_inference_bc_overlay()
        boundary_nodes = result['prob']['boundary_nodes']
        if not boundary_nodes:
            return
        b_complex = result['b_complex']
        b_DC = result.get('b_DC')
        if b_DC is None:
            b_DC = np.zeros(len(boundary_nodes))
        H = self.mosaic_height
        b_amp_H1 = np.abs(b_complex)
        b_amp_DC = np.abs(b_DC)
        H1_max = max(b_amp_H1.max(), 1e-12)
        DC_max = max(b_amp_DC.max(), 1e-12)
        # ── Points layer ──
        pts_yx = []
        for n in boundary_nodes:
            x = float(self.G.nodes[n]['x'])
            y = float(self.G.nodes[n]['y'])
            pts_yx.append([H - y, x])
        pts_yx = np.asarray(pts_yx)
        # Marker size scales linearly with |b̂_H1|; clamped to a min
        # visible size so zero-forcing nodes are still locatable.
        sizes = 12.0 + 60.0 * (b_amp_H1 / H1_max)
        face_colors = cm.viridis(b_amp_DC / DC_max)
        self._inference_bc_layer = self.viewer.add_points(
            pts_yx,
            size=sizes,
            face_color=face_colors,
            border_color='black',
            border_width=0.15,
            opacity=0.92,
            name=(f"Inference BC tile {result['tile_id']}  "
                  f"(size=|b̂_H₁|≤{H1_max:.2f}Pa, "
                  f"color=|b̂_DC − mean|≤{DC_max:.2f}Pa)"),
        )
        # ── Vectors layer for phase arrows ──
        n_b = len(boundary_nodes)
        # Pixel scale for the arrows: pick a fraction of the mosaic span
        # so they're always visible at the default zoom.
        try:
            span = max(self.mosaic_height, self.mosaic_width)
        except Exception:
            span = 1000
        arrow_max_len = 0.025 * float(span)
        vectors = np.zeros((n_b, 2, 2))
        for k, n in enumerate(boundary_nodes):
            x = float(self.G.nodes[n]['x'])
            y = float(self.G.nodes[n]['y'])
            ang = float(np.angle(b_complex[k]))
            scale = float(b_amp_H1[k] / H1_max)
            L_arrow = arrow_max_len * scale
            # napari uses (row, col) with row = down.  Mosaic flips with
            # row = H − y_graph, so increasing y_graph maps to decreasing
            # row.  Math direction (cos ang, sin ang) in (x, y) maps to
            # napari direction (−sin ang, cos ang) in (row, col).
            vectors[k, 0] = [H - y, x]
            vectors[k, 1] = [-L_arrow * np.sin(ang),
                              L_arrow * np.cos(ang)]
        try:
            self._inference_bc_arrow_layer = self.viewer.add_vectors(
                vectors,
                edge_color='red',
                edge_width=2.4,
                opacity=0.95,
                length=1.0,
                vector_style='arrow',  # napari ≥ 0.5
                name=f"Inference BC phase tile {result['tile_id']} (arrow = arg b̂_H₁)",
            )
        except TypeError:
            # napari < 0.5: vector_style not supported; fall back to plain lines.
            self._inference_bc_arrow_layer = self.viewer.add_vectors(
                vectors,
                edge_color='red', edge_width=2.4,
                opacity=0.95, length=1.0,
                name=f"Inference BC phase tile {result['tile_id']} (line = arg b̂_H₁)",
            )


    def _tear_down_inference_bc_overlay(self):
        for attr in ('_inference_bc_layer',
                     '_inference_bc_arrow_layer'):
            lyr = getattr(self, attr, None)
            if lyr is None:
                continue
            try:
                self.viewer.layers.remove(lyr)
            except (KeyError, ValueError):
                pass
            setattr(self, attr, None)


    def _refresh_inference_fisher_overlay(self, result):
        """Per-edge Fisher information about D₀, rendered as a napari
        Shapes (lines) layer where each carve edge's color encodes
        I_j (in log-D units, dimensionless), and a separate Shapes
        layer with the top-K most-informative edges drawn in bold red.

        Reading: bright edges (top of cmap) are doing the work — they're
        the ones whose data is constraining D̂.  Dim edges contribute
        little to the Bayesian update from prior to posterior.

        The TOP-K layer (default K = 10) overlays a thicker red highlight
        on the dominant edges so they pop out at any zoom."""
        from matplotlib import cm, colors as mcolors
        self._tear_down_inference_fisher_overlay()
        prob = result['prob']
        edges_in = prob['edges_in']
        fpe = result.get('fisher_per_edge_logD')
        if fpe is None or len(edges_in) == 0:
            return
        fpe = np.asarray(fpe)
        I_total = max(float(fpe.sum()), 1e-30)
        # ── Edge segments in napari (row, col) coords ──
        H = self.mosaic_height
        lines = []; values = []
        for k, (u, v) in enumerate(edges_in):
            try:
                x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
                x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            except KeyError:
                continue
            lines.append(np.array([[H - y1, x1], [H - y2, x2]]))
            values.append(fpe[k])
        if not lines:
            return
        values = np.asarray(values, dtype=float)
        # Color by log10(I_j) so the wide dynamic range is legible.
        # Floor at the smallest positive value so log10 doesn't explode.
        v_pos = values[values > 0]
        if v_pos.size == 0:
            return
        v_min_pos = max(np.percentile(v_pos, 5), 1e-30)
        v_eff = np.where(values > 0, values, v_min_pos)
        log_v = np.log10(v_eff)
        log_v_clip = np.clip(log_v, np.percentile(log_v, 5),
                              np.percentile(log_v, 95))
        norm = mcolors.Normalize(vmin=log_v_clip.min(),
                                  vmax=log_v_clip.max())
        colors_rgba = cm.viridis(norm(log_v_clip))
        max_frac = float(values.max() / I_total)
        self._inference_fisher_layer = self.viewer.add_shapes(
            lines, shape_type='line',
            edge_color=colors_rgba,
            edge_width=2.4,
            opacity=0.85,
            name=(f"Inference Fisher info tile {result['tile_id']}  "
                  f"(viridis = log₁₀ I_j;  top edge = "
                  f"{max_frac*100:.1f}% of I_total)"),
        )
        # ── Top-K highlight (red, thicker) ──
        K = min(10, len(lines))
        order = np.argsort(-values)
        top_lines = [lines[i] for i in order[:K]]
        top_frac = float(values[order[:K]].sum() / I_total)
        self._inference_fisher_top_layer = self.viewer.add_shapes(
            top_lines, shape_type='line',
            edge_color='red', edge_width=4.5,
            opacity=0.85,
            name=(f"Inference Fisher TOP-{K} tile {result['tile_id']}  "
                  f"(carries {top_frac*100:.0f}% of I_total)"),
        )


    def _tear_down_inference_fisher_overlay(self):
        for attr in ('_inference_fisher_layer',
                     '_inference_fisher_top_layer'):
            lyr = getattr(self, attr, None)
            if lyr is None:
                continue
            try:
                self.viewer.layers.remove(lyr)
            except (KeyError, ValueError):
                pass
            setattr(self, attr, None)


    def _on_inference_run_clicked(self):
        from qtpy.QtWidgets import QApplication
        import time
        tile_id = self._inf_tile_combo.currentData()
        if tile_id is None:
            return
        flat_b_prior = bool(self._inf_flat_b_prior_cb.isChecked())
        # σ_b = 10⁴ Pa is far above the BC confound regime (~50 Pa) and
        # far above the σ-sweep range that's been validated (3–100 Pa).
        # In this limit P₀ ≫ Σ_obs everywhere, so the b prior contribution
        # to C = Σ_obs + T·P₀·Tᵀ is dominated by the data via T·Tᵀ —
        # i.e. an effective improper-flat-prior result with no
        # normalisation pathology, since the marginal log p(y|D, s²)
        # uses the proper Gaussian formula throughout.
        if flat_b_prior:
            sigma_b = 1.0e4
        else:
            sigma_b = float(self._inf_sigma_b_spin.value())
        alpha = float(self._inf_alpha_spin.value())
        bc_length = float(self._inf_bc_length_spin.value())
        # Harmonic mode selector
        harm_idx = self._inf_harmonic_combo.currentIndex()
        harmonic_mode = ['H1', 'H2', 'H1+H2'][harm_idx]
        use_snr_floor = bool(getattr(self, '_inf_snr_floor_cb', None)
                              and self._inf_snr_floor_cb.isChecked())
        self._inf_run_btn.setEnabled(False)
        self._inf_diag_btn.setEnabled(False)
        self._inf_compare_btn.setEnabled(False)
        bc_tag = (f",  ℓ = {bc_length:.0f} px (smooth BC prior)"
                  if bc_length > 0 else "")
        sb_tag = ("σ_b = flat (10⁴ Pa)" if flat_b_prior
                    else f"σ_b = {sigma_b:.1f} Pa")
        self._inf_status_lbl.setText(
            f"<small>Running additive empirical-σ inference on tile "
            f"{tile_id}  ({sb_tag},  α = {alpha:+.2f},  "
            f"harmonic = {harmonic_mode}{bc_tag}) …</small>")
        QApplication.processEvents()
        t0 = time.time()
        try:
            res = self._inference_run(int(tile_id),
                                         sigma_b_Pa=sigma_b,
                                         alpha=alpha,
                                         bc_length_px=bc_length,
                                         harmonic_mode=harmonic_mode,
                                         use_snr_floor=use_snr_floor)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._inf_status_lbl.setText(
                f"<small><font color='#c33'>Inference failed: "
                f"{type(e).__name__}: {e}</font></small>")
            self._inf_run_btn.setEnabled(True)
            return
        elapsed = time.time() - t0
        if res.get('error'):
            self._inf_status_lbl.setText(
                f"<small><font color='#c33'>{res['error']}</font></small>")
            self._inf_run_btn.setEnabled(True)
            return
        # Format result summary
        prob = res['prob']
        n_edges = len(prob['edges_in'])
        n_valid_h1 = int((prob['valid_dc'] & prob['valid_h1']).sum())
        n_B = len(prob['boundary_nodes'])
        bc_len = res.get('bc_length_px', 0.0)
        bc_str = (f"  ·  ℓ = {bc_len:.0f} px (smooth BC)"
                  if bc_len > 0 else "  ·  ℓ = 0  (independent BCs)")
        harm_str = f"  ·  harmonic = {res.get('harmonic_mode', 'H1')}"
        lines = [
            f"Tile {res['tile_id']}  ·  α = {res['alpha']:+.2f}  ·  "
            f"σ_b = {res['sigma_b']:.1f} Pa{bc_str}{harm_str}",
            "",
            f"  D̂           = {res['D_hat']:.3e}  /Pa "
            f"(at R_ref = {res['R_ref_um']:.1f} µm)",
            f"  3-nat support region = [{res['lo']:.2e}, {res['hi']:.2e}]",
            f"  90% HPD CI            = [{res.get('lo_90', float('nan')):.2e}, "
            f"{res.get('hi_90', float('nan')):.2e}]  "
            f"(width {res.get('width_90', float('nan')):.2f} dec)",
            f"  CI width    = {res['width']:.2f} decades  ·  {res['tier']}",
            "",
            # Final IDENTIFICATION verdict: simple, single threshold on Δ@0
            f"  Δ@0          = {res['delta_at_zero']:+.1f} nats  "
            f"({'IDENTIFIED' if res['delta_at_zero'] >= 10 else 'not identified'})",
            "",
            "  ── Inferred noise model:  σ²_j = â² + ŝ²·|Q_j|² ──",
            f"  â  (measurement floor)     = {res['a_at_map']*1e12:.4f} nL/s",
            f"  ŝ  (proportional misfit)   = {res['s_at_map']:.3f}  "
            f"(dimensionless)",
            f"  |Q|* = â / ŝ (crossover)   = {res['Q_crossover']*1e12:.4f} nL/s",
            "           below |Q|* = measurement-noise-dominated edges",
            "",
            f"  n_meas       = {n_valid_h1}  (of {n_edges} carve edges)",
            (f"  n_boundary   = {n_B}  "
             f"({res.get('n_av_promoted', 0)} A/V anchors promoted from interior)"
             if res.get('n_av_promoted', 0)
             else f"  n_boundary   = {n_B}"),
            f"  median SNR  = {res['snr_median']:+.1f} dB",
            f"  R range     = [{res['R_min_um']:.1f}, "
            f"{res['R_max_um']:.1f}] µm (median {res['R_ref_um']:.1f})",
        ]
        # ── Bayesian information summary (per-edge Fisher info) ──
        fpe_logD = res.get('fisher_per_edge_logD')
        if fpe_logD is not None:
            fpe_logD = np.asarray(fpe_logD)
            I_total = float(fpe_logD.sum())
            prior_prec = 1.0 / (np.log(10.0) ** 2)
            ratio = I_total / max(prior_prec, 1e-30)
            n_eff = (fpe_logD.sum() ** 2) / max((fpe_logD ** 2).sum(), 1e-30)
            sigma_logD = 1.0 / np.sqrt(max(prior_prec + I_total, 1e-30))
            D_lap_lo = res['D_hat'] * np.exp(-1.96 * sigma_logD)
            D_lap_hi = res['D_hat'] * np.exp(+1.96 * sigma_logD)
            lap_w = (np.log10(D_lap_hi / D_lap_lo)
                     if D_lap_lo > 0 else float('inf'))
            # Top-K contributions
            top_k = min(5, fpe_logD.size)
            order = np.argsort(-fpe_logD)
            top_frac = (fpe_logD[order[:top_k]].sum()
                        / max(I_total, 1e-30))
            lines.append("")
            lines.append(
                "  ── Bayesian information (Fisher / Laplace) ──")
            lines.append(
                f"  I_total (log-D)            = {I_total:.2f}")
            lines.append(
                f"  prior precision (1/τ_D²)  = {prior_prec:.3f}")
            lines.append(
                f"  data/prior ratio           = {ratio:.1f}× "
                f"({'data-dominated' if ratio >= 5 else 'mixed' if ratio >= 2 else 'PRIOR-DRIVEN'})")
            lines.append(
                f"  n_effective informative   = {n_eff:.1f}  "
                f"(of {n_valid_h1} measured edges)")
            lines.append(
                f"  top-{top_k} edges contribute {top_frac*100:.0f}% of total Fisher info")
            lines.append(
                f"  Laplace 95% CI              = [{D_lap_lo:.2e}, "
                f"{D_lap_hi:.2e}]  (width {lap_w:.2f} dec)")
        self._inf_results_box.setPlainText("\n".join(lines))
        self._inf_status_lbl.setText(
            f"<small>OK · {elapsed:.1f}s · D̂ = {res['D_hat']:.2e} /Pa.  "
            f"Click <b>D₀ posterior</b> for the evidence plot, or pick an "
            f"edge in the viewer then <b>Compare</b> for predicted vs "
            f"measured Q(t).</small>")
        self._inference_last_result = (int(tile_id), res)
        self._inf_run_btn.setEnabled(True)
        self._inf_diag_btn.setEnabled(True)
        self._inf_compare_btn.setEnabled(True)
        if not self._inf_show_carve_chk.isChecked():
            self._inf_show_carve_chk.setChecked(True)
        # Draw the inferred boundary forcing on the mosaic as napari
        # layers — Points (size = |b̂_H₁|, color = |b̂_DC|) + Vectors
        # (arrows showing arg(b̂_H₁)).  Persistent on the canvas until
        # the next inference run or a tile change.
        try:
            self._refresh_inference_bc_overlay(res)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._inf_status_lbl.setText(
                self._inf_status_lbl.text() +
                f"  <font color='#c33'>(BC overlay failed: "
                f"{type(e).__name__}: {e})</font>")
        # Per-edge Fisher-info / sensitivity overlay
        try:
            self._refresh_inference_fisher_overlay(res)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._inf_status_lbl.setText(
                self._inf_status_lbl.text() +
                f"  <font color='#c33'>(Fisher overlay failed: "
                f"{type(e).__name__}: {e})</font>")


    def _on_inference_diagnostic_clicked(self):
        from qtpy.QtWidgets import QApplication
        if self._inference_last_result is None:
            return
        tile_id, res = self._inference_last_result
        self._inf_status_lbl.setText(
            "<small>Rendering D₀ posterior …</small>")
        QApplication.processEvents()
        try:
            self._inference_render_diagnostic(tile_id, res)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._inf_status_lbl.setText(
                f"<small><font color='#c33'>Diagnostic render failed: "
                f"{type(e).__name__}: {e}</font></small>")
            return
        self._inf_status_lbl.setText(
            f"<small>D₀ posterior for tile {tile_id} rendered.</small>")


    def _on_inference_compare_clicked(self):
        """Compare measured vs predicted Q(t) for the currently-selected
        edge, using the MAP D̂ + b̂ from the last inference run."""
        from qtpy.QtWidgets import QApplication
        if self._inference_last_result is None:
            self._inf_status_lbl.setText(
                "<small><font color='#c33'>No inference result — run "
                "the inference first.</font></small>")
            return
        if self._selected_edge is None:
            self._inf_status_lbl.setText(
                "<small><font color='#c33'>No edge selected — click an "
                "edge in the viewer first (the Time-series tab updates "
                "the selection).</font></small>")
            return
        tile_id, res = self._inference_last_result
        u, v = self._selected_edge
        # Look up edge in the carve's edges_in
        edges_in = res['prob']['edges_in']
        edge_idx = None
        for i, (a, b) in enumerate(edges_in):
            if (a, b) == (u, v) or (b, a) == (u, v):
                edge_idx = i; break
        if edge_idx is None:
            self._inf_status_lbl.setText(
                f"<small><font color='#c33'>Selected edge ({u}, {v}) "
                f"is not in tile {tile_id}'s carve.</font></small>")
            return
        self._inf_status_lbl.setText(
            f"<small>Rendering compare plot for edge ({u}, {v}) "
            f"in tile {tile_id} …</small>")
        QApplication.processEvents()
        try:
            self._inference_render_compare(tile_id, res, edge_idx)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._inf_status_lbl.setText(
                f"<small><font color='#c33'>Compare render failed: "
                f"{type(e).__name__}: {e}</font></small>")
            return
        self._inf_status_lbl.setText(
            f"<small>Compare plot for edge ({u}, {v}) rendered.</small>")


    def _inference_render_compare(self, tile_id, result, edge_idx):
        """Three-panel comparison for one edge:
          (1) Phasor — measured H₁ vs predicted H₁ (complex plane),
              with the per-edge σ_meas as a circle around the measurement.
          (2) Single-cycle overlay — measured Q(t) cycle-folded (binned)
              vs reconstructed Q_DC + |H₁|·cos(ωt + φ) at MAP.
          (3) Full raw Q(t) — measured vs reconstructed (DC + H₁).
        """
        import matplotlib.pyplot as plt
        import tempfile
        prob = result['prob']
        edges_in = prob['edges_in']
        u, v = edges_in[edge_idx]
        ed = self.G.edges[u, v]
        # Pick the per-tile measurement for this edge
        m = next((mm for mm in (ed.get('measurements_piv') or [])
                   if MeasurementView(mm).tile_id == int(tile_id)), None)
        mv = MeasurementView(m) if m is not None else None
        # v4 migration: was m.get('Q_t')
        if m is None or mv.Q_t is None:
            raise RuntimeError(
                f"Edge ({u}, {v}) has no Q_t measurement on tile {tile_id}")
        # Canonical convention: m['Q_t'], m['Q_DC'], m['Q_H1_amp'] are
        # stored in the FLOW_FROM → FLOW_TO direction.  The model
        # works in the (u, v) graph-storage order.  When flow_from→
        # flow_to is opposite (u, v), apply `sign = −1` to bring the
        # measurement into the model's convention so the comparison
        # is apples-to-apples.
        # ed.get('flow_from')/('flow_to') is an edge-level attr, not the
        # _piv-promoted one; kept.
        ff, ft = ed.get('flow_from'), ed.get('flow_to')
        sign = 1.0 if (ff == u and ft == v) else -1.0
        Q_t = np.asarray(mv.Q_t, dtype=float) * sign  # v4: was m['Q_t']
        # f₀: prefer m's harmonic-fit frequency (matches the canonical
        # Q_H1_amp / Q_H1_phi); fall back to tile-level f₀ then 2.5.
        tile_f0_map = self.G.graph.get('tile_f0_piv') or {}
        # v4 migration: was m.get('f0_hz')
        f0 = float(mv.harm.f0_hz
                   or tile_f0_map.get(int(tile_id))
                   or self.G.graph.get('f0_hz')
                   or 2.5)
        dt = 1.0 / 250.0
        N = len(Q_t)
        t_axis = np.arange(N) * dt
        amp = float(m.get('Q_H1_amp', 0.0))
        phi = float(m.get('Q_H1_phi', 0.0))
        Q_DC_meas_nL = float(m.get('Q_DC', 0.0)) * sign
        Q_H1_meas_nL = amp * np.exp(1j * phi) * sign
        # H₂ (if available on this edge)
        amp_H2 = m.get('Q_H2_amp', None)
        phi_H2 = m.get('Q_H2_phi', None)
        if (amp_H2 is not None and phi_H2 is not None
                and np.isfinite(amp_H2) and np.isfinite(phi_H2)):
            amp_H2 = float(amp_H2); phi_H2 = float(phi_H2)
            Q_H2_meas_nL = amp_H2 * np.exp(1j * phi_H2) * sign
            has_H2 = True
        else:
            amp_H2 = float('nan'); phi_H2 = float('nan')
            Q_H2_meas_nL = complex('nan')
            has_H2 = False
        SCALE = 1e12   # m³/s → nL/s
        # Inferred Q̄ from the DC solve (b̂_DC).  Q_DC_pred_full is in m³/s
        # in the (u,v) edge-orientation convention used by the inference;
        # convert to nL/s and back into the canonical (flow_from→flow_to)
        # convention so it lines up with Q_DC_meas_nL above.
        Q_DC_pred_full = result.get('Q_DC_pred_full')
        if Q_DC_pred_full is not None and edge_idx < len(Q_DC_pred_full):
            Q_DC_pred_nL = float(Q_DC_pred_full[edge_idx]) * SCALE * sign
        else:
            Q_DC_pred_nL = float('nan')
        Q_H1_pred_nL = result['Q_H1_pred'][edge_idx] * SCALE
        # Predicted H₂ phasor — only populated in H₁+H₂ joint mode
        Q_H2_pred_arr = result.get('Q_H2_pred')
        if (Q_H2_pred_arr is not None
                and edge_idx < len(Q_H2_pred_arr)
                and np.all(np.isfinite([Q_H2_pred_arr[edge_idx].real,
                                          Q_H2_pred_arr[edge_idx].imag]))):
            Q_H2_pred_nL = complex(Q_H2_pred_arr[edge_idx]) * SCALE
            has_H2_pred = True
        else:
            Q_H2_pred_nL = complex('nan')
            has_H2_pred = False
        # Per-edge σ for H₁ under the inferred additive+proportional model:
        #   σ²_total = â² + ŝ²·|Q|²
        # â is the absolute measurement floor (in m³/s); ŝ² is the
        # proportional misfit.  In display units (nL/s):
        a_nL = result['a_at_map'] * SCALE         # m³/s → nL/s
        s_map = result['s_at_map']
        sigma_H1_total_nL = np.sqrt(
            a_nL**2 + (s_map * abs(Q_H1_meas_nL))**2)
        # For reference: the original SNR-derived measurement σ.  This is
        # no longer used in the inference (γ → 0 from the inferred-shape
        # test), but display it alongside for diagnostic.
        snr_h1_db = m.get('Q_H1_snr_db', np.nan)
        if np.isfinite(snr_h1_db):
            sigma_H1_meas_nL = abs(Q_H1_meas_nL) * 10.0 ** (-snr_h1_db / 20.0)
        else:
            sigma_H1_meas_nL = np.nan
        sigma_H1_nL = sigma_H1_meas_nL   # kept for downstream lim computation
        # ── Figure ──
        fig = plt.figure(figsize=(14, 14), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0])
        # Phasor panel
        ax = fig.add_subplot(gs[0, 0])
        # Measurement with σ circle
        ax.plot([0, Q_H1_meas_nL.real], [0, Q_H1_meas_nL.imag],
                 color='C0', lw=2.0, label=f'measured |H₁| = {abs(Q_H1_meas_nL):.3f} nL/s')
        ax.plot(Q_H1_meas_nL.real, Q_H1_meas_nL.imag, 'o',
                 color='C0', ms=10)
        theta = np.linspace(0, 2*np.pi, 80)
        if np.isfinite(sigma_H1_meas_nL) and sigma_H1_meas_nL > 0:
            ax.plot(Q_H1_meas_nL.real + sigma_H1_meas_nL * np.cos(theta),
                     Q_H1_meas_nL.imag + sigma_H1_meas_nL * np.sin(theta),
                     color='C0', lw=0.8, ls=':', alpha=0.7,
                     label=f'σ_meas (PIV noise) = {sigma_H1_meas_nL:.3f} nL/s '
                            f'(SNR {snr_h1_db:.1f} dB)')
        if np.isfinite(sigma_H1_total_nL) and sigma_H1_total_nL > 0:
            ax.plot(Q_H1_meas_nL.real + sigma_H1_total_nL * np.cos(theta),
                     Q_H1_meas_nL.imag + sigma_H1_total_nL * np.sin(theta),
                     color='C0', lw=1.2, ls='--', alpha=0.6,
                     label=f'σ_total (meas + ŝ²·|Q|) = '
                            f'{sigma_H1_total_nL:.3f} nL/s')
        # Prediction
        ax.plot([0, Q_H1_pred_nL.real], [0, Q_H1_pred_nL.imag],
                 color='C3', lw=2.0,
                 label=f'predicted |H₁| = {abs(Q_H1_pred_nL):.3f} nL/s')
        ax.plot(Q_H1_pred_nL.real, Q_H1_pred_nL.imag, '^',
                 color='C3', ms=10)
        # H₂ phasors (when available) — drawn fainter / dotted to keep
        # the H₁ phasor as the dominant visual.
        if has_H2:
            ax.plot([0, Q_H2_meas_nL.real], [0, Q_H2_meas_nL.imag],
                     color='C0', lw=1.0, ls=':', alpha=0.7,
                     label=f'measured |H₂| = {abs(Q_H2_meas_nL):.3f} nL/s')
            ax.plot(Q_H2_meas_nL.real, Q_H2_meas_nL.imag, 's',
                     color='C0', ms=7, mfc='none', mew=1.5)
        if has_H2_pred:
            ax.plot([0, Q_H2_pred_nL.real], [0, Q_H2_pred_nL.imag],
                     color='C3', lw=1.0, ls=':', alpha=0.7,
                     label=f'predicted |H₂| = {abs(Q_H2_pred_nL):.3f} nL/s')
            ax.plot(Q_H2_pred_nL.real, Q_H2_pred_nL.imag, 's',
                     color='C3', ms=7, mfc='none', mew=1.5)
        # Origin marker
        ax.plot(0, 0, 'k+', ms=8)
        lim_candidates = [abs(Q_H1_meas_nL), abs(Q_H1_pred_nL),
                            sigma_H1_total_nL
                            if np.isfinite(sigma_H1_total_nL) else 0,
                            sigma_H1_meas_nL
                            if np.isfinite(sigma_H1_meas_nL) else 0]
        if has_H2:    lim_candidates.append(abs(Q_H2_meas_nL))
        if has_H2_pred: lim_candidates.append(abs(Q_H2_pred_nL))
        lim = 1.4 * max(lim_candidates)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel('Re Q_H₁  (nL/s)')
        ax.set_ylabel('Im Q_H₁  (nL/s)')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.axhline(0, color='#888', lw=0.5)
        ax.axvline(0, color='#888', lw=0.5)
        dphi_deg = np.degrees(np.angle(Q_H1_pred_nL) -
                                np.angle(Q_H1_meas_nL))
        ax.set_title(
            f"H₁ phasor — measured vs predicted\n"
            f"|pred|/|meas| = {abs(Q_H1_pred_nL)/max(abs(Q_H1_meas_nL),1e-12):.2f}  ·  "
            f"Δφ = {((dphi_deg + 180) % 360) - 180:+.1f}°",
            fontsize=10)
        ax.legend(fontsize=8, loc='upper right')

        # Cycle-folded overlay
        # ── Carve + boundary-node H₁ forcing ──
        # Spatial map of the inference carve.  Edges in light grey;
        # selected edge in red.  At each boundary node:
        #   color  → |Q_DC| throughput  (sum of measured |Q_DC| on
        #            carve edges adjacent to the BC node, nL/s).
        #            Tells you which boundary nodes are "DC-busy" —
        #            i.e. carry meaningful flow.
        #   size   → |b̂_n|  amplitude of the inferred H₁ pressure
        #            forcing at that BC node, Pa.
        #   arrow  → arg(b̂_n)  the H₁ phase, drawn as an unambiguous
        #            arrow (one direction only — no 180° flip).
        ax = fig.add_subplot(gs[0, 1])
        prob = result['prob']
        boundary_nodes = prob['boundary_nodes']
        edges_in = prob['edges_in']
        b_complex = result['b_complex']
        Q_DC_obs = prob['Q_DC_obs']           # SI m³/s (signed)
        valid_dc = prob['valid_dc']
        # Gather node positions for all nodes touched by the carve
        all_nodes_xy = {}
        for (uu, vv) in edges_in:
            for nn in (uu, vv):
                if nn not in all_nodes_xy:
                    nd = self.G.nodes[nn]
                    all_nodes_xy[nn] = (float(nd.get('x', 0)),
                                          float(nd.get('y', 0)))
        # Draw all carve edges (light grey, selected edge highlighted)
        sel_drawn = False
        for (uu, vv) in edges_in:
            x1, y1 = all_nodes_xy[uu]; x2, y2 = all_nodes_xy[vv]
            is_selected = ((uu, vv) == (u, v) or (vv, uu) == (u, v))
            if is_selected:
                ax.plot([x1, x2], [y1, y2], color='red',
                         lw=3.5, zorder=4, alpha=0.95,
                         label='selected edge' if not sel_drawn else None)
                sel_drawn = True
            else:
                ax.plot([x1, x2], [y1, y2], color='#bbb',
                         lw=0.6, alpha=0.6, zorder=1)
        # DC throughput per BC node: sum of |Q_DC| over carve edges
        # adjacent to the BC node, converted to nL/s for display.
        SCALE_nL = 1e12
        boundary_set = set(boundary_nodes)
        node_DC_through = {n: 0.0 for n in boundary_nodes}
        for i, (uu, vv) in enumerate(edges_in):
            if not valid_dc[i]: continue
            q = abs(Q_DC_obs[i]) * SCALE_nL
            if uu in boundary_set: node_DC_through[uu] += q
            if vv in boundary_set: node_DC_through[vv] += q
        DC_values = np.array([node_DC_through[n] for n in boundary_nodes])
        # b̂ amplitude + phase
        b_amp = np.abs(b_complex)
        amp_max = max(b_amp.max(), 1e-12)
        bx = np.array([all_nodes_xy[n][0] for n in boundary_nodes])
        by = np.array([all_nodes_xy[n][1] for n in boundary_nodes])
        # Marker size ∝ |b̂|.  Note: marker AREA scales with the 's'
        # arg in scatter, so to make perceived diameter ∝ |b̂| we want
        # s ∝ |b̂|² .  We compromise with s ∝ |b̂| so even small forcings
        # remain visible.
        sizes = 40 + 350.0 * (b_amp / amp_max)
        # Color from DC throughput on a sequential cmap
        sc = ax.scatter(bx, by, s=sizes, c=DC_values, cmap='viridis',
                         edgecolor='black', linewidth=0.7, zorder=5,
                         alpha=0.95)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02,
                             label='Σ |Q_DC| adjacent (nL/s)')
        # Unambiguous phase arrows.  One direction only — origin at BC
        # node, head at (cos φ, sin φ) — no 180° flip risk.  Length
        # also scaled by |b̂| so small forcings get short arrows.
        ax_span = max(bx.max() - bx.min(), by.max() - by.min(),
                        1.0) if len(bx) else 1.0
        arrow_max_len = 0.06 * ax_span
        for k_node, n in enumerate(boundary_nodes):
            if b_amp[k_node] == 0: continue
            x0, y0 = all_nodes_xy[n]
            ang = np.angle(b_complex[k_node])
            scale = b_amp[k_node] / amp_max
            dx_arrow = arrow_max_len * scale * np.cos(ang)
            dy_arrow = arrow_max_len * scale * np.sin(ang)
            ax.annotate('', xy=(x0 + dx_arrow, y0 + dy_arrow),
                         xytext=(x0, y0),
                         arrowprops=dict(arrowstyle='-|>', color='black',
                                          lw=1.4, alpha=0.9,
                                          mutation_scale=10),
                         zorder=6)
        ax.set_aspect('equal')
        ax.set_xlabel('x (graph px)'); ax.set_ylabel('y (graph px)')
        ax.set_title(
            f"Carve + BC forcing  ({len(boundary_nodes)} BC nodes)\n"
            f"size ∝ |b̂_H₁|  (max = {amp_max:.2f} Pa)  ·  "
            f"arrow ∝ phase  ·  color = ΣDC throughput",
            fontsize=9)
        if sel_drawn:
            ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)

        # ── Single-cycle overlay (was gs[0,1] → now gs[1,:]) ──
        ax = fig.add_subplot(gs[1, :])
        T_period = 1.0 / f0
        # Measured cycle-fold: bin Q_t by cycle phase
        cycle_phase = (t_axis * f0) % 1.0
        N_bins = 32
        bins = np.linspace(0, 1, N_bins + 1)
        binned = np.full(N_bins, np.nan)
        for k in range(N_bins):
            mask = (cycle_phase >= bins[k]) & (cycle_phase < bins[k+1])
            if mask.any():
                binned[k] = np.mean(Q_t[mask])
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax.plot(bin_centers, binned, 'o-', color='C0',
                 lw=1.5, ms=5, label='measured (cycle-folded)')
        # Reconstructed: Q_DC + |H1|·cos(2πf₀t + φ)  [ + H₂ if available ]
        t_one = np.linspace(0, T_period, 200)
        # Measured reconstruction: H₁ alone (and H₁+H₂ if measured)
        Q_recon_meas = Q_DC_meas_nL + amp * np.cos(2*np.pi*f0*t_one + phi) * sign
        ax.plot(t_one / T_period, Q_recon_meas, color='C0',
                 lw=1.0, ls='--', alpha=0.7,
                 label='measured Q̄ + H₁')
        if has_H2:
            Q_recon_meas_H12 = Q_recon_meas + amp_H2 * np.cos(
                2 * np.pi * (2 * f0) * t_one + phi_H2) * sign
            ax.plot(t_one / T_period, Q_recon_meas_H12, color='C0',
                     lw=1.2, alpha=0.85,
                     label='measured Q̄ + H₁ + H₂')
        # Predicted reconstruction: Q̄_pred + H₁_pred (+ H₂_pred if joint)
        Q_pred_dc = Q_DC_pred_nL if np.isfinite(Q_DC_pred_nL) else Q_DC_meas_nL
        Q_recon_pred = Q_pred_dc + abs(Q_H1_pred_nL) * np.cos(
            2*np.pi*f0*t_one + np.angle(Q_H1_pred_nL))
        ax.plot(t_one / T_period, Q_recon_pred, color='C3',
                 lw=1.6, ls='--', alpha=0.85,
                 label='predicted Q̄ + H₁ at MAP')
        if has_H2_pred:
            Q_recon_pred_H12 = Q_recon_pred + abs(Q_H2_pred_nL) * np.cos(
                2 * np.pi * (2 * f0) * t_one + np.angle(Q_H2_pred_nL))
            ax.plot(t_one / T_period, Q_recon_pred_H12, color='C3',
                     lw=2.0,
                     label='predicted Q̄ + H₁ + H₂ at MAP')
        ax.set_xlabel('cycle fraction')
        ax.set_ylabel('Q (nL/s)')
        ax.set_title('Single-cycle: measured vs predicted', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

        # ── Full Q(t) raw ──
        # Show three traces:
        #   - measured Q(t) raw
        #   - measured reconstruction = Q̄_meas + |H₁|_meas·cos(...)
        #   - PREDICTED reconstruction = Q̄_pred + |H₁|_pred·cos(...)
        # where Q̄_pred = T_DC[e]·b̂_DC is the INFERRED DC mean (not the
        # measured one).  Both Q̄_pred and |H₁|_pred come from the same
        # boundary-pressure fit, so the red curve is what the model
        # actually predicts at this edge.  The horizontal dashed lines
        # mark the two DC means side-by-side for visual comparison.
        ax = fig.add_subplot(gs[2, :])
        ax.plot(t_axis, Q_t, color='C0', lw=0.6, alpha=0.85,
                 label='measured Q(t) raw')
        # Measured H1 reconstruction (uses measured Q̄)
        Q_meas_recon = Q_DC_meas_nL + amp * np.cos(
            2*np.pi*f0*t_axis + phi) * sign
        ax.plot(t_axis, Q_meas_recon, color='C0', lw=1.2, ls=':',
                 label=f'measured Q̄ + H₁ fit  (Q̄_meas = {Q_DC_meas_nL:+.4f} nL/s)')
        # Measured H₁+H₂ reconstruction — visible when H₂ is measured
        if has_H2:
            Q_meas_recon_H12 = Q_meas_recon + amp_H2 * np.cos(
                2 * np.pi * (2 * f0) * t_axis + phi_H2) * sign
            ax.plot(t_axis, Q_meas_recon_H12, color='C0', lw=1.3, ls='-',
                     alpha=0.85,
                     label=f'measured Q̄ + H₁ + H₂  (|H₂| = {amp_H2:.4f} nL/s)')
        # PREDICTED reconstruction: inferred Q̄ + inferred H1 (+ H₂ in joint)
        Q_pred_full = Q_DC_pred_nL + abs(Q_H1_pred_nL) * np.cos(
            2*np.pi*f0*t_axis + np.angle(Q_H1_pred_nL))
        ax.plot(t_axis, Q_pred_full, color='C3', lw=1.4, ls='--', alpha=0.85,
                 label=f'predicted Q̄ + H₁ at MAP  (Q̄_pred = {Q_DC_pred_nL:+.4f} nL/s)')
        if has_H2_pred:
            Q_pred_full_H12 = Q_pred_full + abs(Q_H2_pred_nL) * np.cos(
                2 * np.pi * (2 * f0) * t_axis + np.angle(Q_H2_pred_nL))
            ax.plot(t_axis, Q_pred_full_H12, color='C3', lw=1.8,
                     label=f'predicted Q̄ + H₁ + H₂ at MAP  '
                            f'(|H₂_pred| = {abs(Q_H2_pred_nL):.4f} nL/s)')
        # DC reference lines
        ax.axhline(Q_DC_meas_nL, color='C0', ls='-', lw=0.6, alpha=0.4)
        if np.isfinite(Q_DC_pred_nL):
            ax.axhline(Q_DC_pred_nL, color='C3', ls='-', lw=0.6, alpha=0.4)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('Q (nL/s)')
        # Title shows the DC residual at this edge
        if np.isfinite(Q_DC_pred_nL):
            dc_resid = Q_DC_meas_nL - Q_DC_pred_nL
            ax.set_title(f'Full Q(t)  ·  ΔQ̄ (meas − pred) = '
                         f'{dc_resid:+.4f} nL/s', fontsize=10)
        else:
            ax.set_title('Full Q(t)', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Tile {tile_id}, edge ({u}, {v})  —  "
            f"D̂ = {result['D_hat']:.2e} /Pa  ·  "
            f"α = {result['alpha']:+.2f}  ·  f₀ = {f0:.2f} Hz",
            fontsize=11, fontweight='bold')

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as _tmp:
            _tmp_path = _tmp.name
        fig.savefig(_tmp_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        self._show_diagnostic_png(_tmp_path, int(tile_id), result)
