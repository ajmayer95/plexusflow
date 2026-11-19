"""Time-series tab: vessel comparison set and measured / reconciled / simulated Q(t) plots.

Mixin of plexusflow.viewer.app.ReadOnlyMosaicViewer (methods moved verbatim from app.py)."""
from __future__ import annotations

from ._common import (  # noqa: F401
    MeasurementView,
    QUALITY_RANK,
    _best_measurement,
    _harmonic_class,
    _measurement_usable,
    _place_figure_top_right,
    np,
    plt,
)


class TimeseriesMixin:

    # ── comparison set management ──────────────────────────────────────
    def _add_to_comparison(self):
        if self._selected_edge is None:
            return
        u, v = self._selected_edge
        if (u, v) in self._comparison_edges:
            return  # already in set
        self._comparison_edges.append((u, v))
        self._refresh_comparison_markers()
        self._refresh_comparison_buttons()


    def _clear_comparison(self):
        self._comparison_edges.clear()
        self._refresh_comparison_markers()
        self._refresh_comparison_buttons()


    def _refresh_comparison_markers(self):
        # Always tear down the previous edge overlay — napari Shapes
        # layers don't always repaint cleanly on .data/.edge_color reassign
        # (same bug worked around in the main edges layer rebuild).
        if self._comparison_edge_overlay_layer is not None:
            try:
                self.viewer.layers.remove(
                    self._comparison_edge_overlay_layer)
            except (KeyError, ValueError):
                pass
            self._comparison_edge_overlay_layer = None

        if self._comparison_marker_layer is None:
            return
        if not self._comparison_edges:
            self._comparison_marker_layer.data = np.empty((0, 2))
            return
        # Match the comparison plot's per-edge colors:
        #   index 0 → matplotlib C0 (blue, Edge A)
        #   index 1 → matplotlib C1 (orange, Edge B)
        # Any extras beyond the two used by the plot fade to grey.
        comp_colors = ['#1f77b4', '#ff7f0e']
        H = self.mosaic_height
        pts, marker_colors = [], []
        lines, line_colors, line_widths = [], [], []
        for k, (u, v) in enumerate(self._comparison_edges):
            try:
                x1, y1 = self.G.nodes[u]['x'], self.G.nodes[u]['y']
                x2, y2 = self.G.nodes[v]['x'], self.G.nodes[v]['y']
            except KeyError:
                continue
            color = (comp_colors[k] if k < len(comp_colors)
                     else '#888888')
            # Mid-point dot for the small marker layer.
            pts.append([H - 0.5 * (y1 + y2), 0.5 * (x1 + x2)])
            marker_colors.append(color)
            # Thick bright line overlaying the actual edge — the
            # primary "this edge is in the comparison set" signal,
            # visible on top of any field colouring.
            lines.append(np.array([[H - y1, x1], [H - y2, x2]]))
            line_colors.append(color)
            # Width is generous; the underlying edge_width caps at
            # EDGE_WIDTH_MAX=22 so 16 px sits well above any vessel
            # while not swamping the image.
            line_widths.append(16.0)

        # Small filled dot at each edge midpoint — secondary cue, easy
        # to see when the edge itself is short or off-screen.
        self._comparison_marker_layer.data = np.asarray(pts)
        self._comparison_marker_layer.size = 14
        try:
            self._comparison_marker_layer.border_color = marker_colors
            self._comparison_marker_layer.face_color = marker_colors
        except (AttributeError, TypeError):
            try:
                self._comparison_marker_layer.edge_color = marker_colors
                self._comparison_marker_layer.face_color = marker_colors
            except (AttributeError, TypeError):
                pass

        # Bold edge overlay — the headline marker.
        if lines:
            self._comparison_edge_overlay_layer = self.viewer.add_shapes(
                lines, shape_type='line',
                edge_color=line_colors,
                edge_width=line_widths,
                name='Comparison edges',
                opacity=0.85,
            )


    def _refresh_comparison_buttons(self):
        n = len(self._comparison_edges)
        self._plot_comp_btn.setText(f"Plot comparison ({n})")
        self._plot_comp_btn.setEnabled(n >= 2)
        self._clear_comp_btn.setEnabled(n > 0)


    def _build_sim_measurement(self, u, v, n_frames: int = 750,
                                quantity: str = 'Q'):
        """Reconstruct a measurement-shaped dict from the stored
        per-edge sim harmonics for edge (u, v).

        `quantity = 'Q'` pulls from `_sim_tmp_harmonics` (flow); `'P'`
        pulls from `_sim_tmp_p_harmonics` (midpoint pressure).  The
        returned dict has the same keys the comparison plot reads from
        `measurements_piv` (Q_t, f0_hz, mean_Q, amp_Q, PI, tile_id),
        so it can flow through the existing pipeline unchanged; the
        Q_t slot carries P(t) values in pressure mode and the caller
        relabels axes accordingly.

        Returns None when the edge has no sim data or f0 is missing.
        """
        if not self.G.has_edge(u, v):
            return None
        d = self.G.edges[u, v]
        harm_key = ('_sim_tmp_p_harmonics' if quantity == 'P'
                    else '_sim_tmp_harmonics')
        harm = d.get(harm_key)
        if harm is None:
            return None
        f0 = d.get('_sim_tmp_f0_hz')
        if f0 is None:
            f0 = self._sim_last_f0_hz
        try:
            f0 = float(f0)
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(f0) and f0 > 0):
            return None
        arr = np.asarray(harm, dtype=complex)
        dt = 1.0 / 250.0
        t = np.arange(n_frames) * dt
        omega = 2 * np.pi * f0
        # Real-time reconstruction: f(t) = Σ_k Re{ c_k · exp(i k ω t) }
        #                                = Σ_k Re(c_k) cos(kωt) − Im(c_k) sin(kωt)
        sig = np.full(n_frames, arr[0].real, dtype=float)
        for k in range(1, len(arr)):
            sig += (arr[k].real * np.cos(k * omega * t)
                    - arr[k].imag * np.sin(k * omega * t))
        mean_val = float(arr[0].real)
        amp_val = float(abs(arr[1])) if len(arr) > 1 else float('nan')
        PI = (2.0 * amp_val / abs(mean_val)
              if abs(mean_val) > 1e-12 else float('nan'))
        return {
            'Q_t': sig,        # carries P(t) when quantity='P'
            'f0_hz': f0,
            'mean_Q': mean_val,
            'amp_Q': amp_val,
            'PI': PI,
            'tile_id': None,
            'quality_tier': 'A',
            'snr_pulse': 99.0,
            'snr_f0': 99.0,
            'fit_success': True,
        }


    def _plot_comparison(self):
        """Two-edge comparison figure focused on harmonic attenuation.

        Layout (three rows):
          Row 1 — two raw Q(t) panels (one per edge), shared y-axis;
                  amplitude attenuation reads visually.
          Row 2 — single-period overlay of bold harmonic fits (no
                  individual cycles); tiny corner note with per-edge Z
                  values for H₁, H₂.
          Row 3 — attenuation–phase scatter, one marker per harmonic.
                  x = Δφ (degrees, B − A); y = pulse-relative
                  attenuation. Fixed axes (−90°…+90°, 0…1.2) for
                  cross-call consistency. Reference lines at y=1, x=0.

        Pulse-relative attenuation (per harmonic k):

            atten_k = (|H_k^B| / Q̄_B) / (|H_k^A| / Q̄_A)

        strips out the trivial mean-flow rescaling and isolates the
        per-harmonic damping of the pulsatile component.  Q̄ is taken
        from the harmonic-fit DC term (a0).

        H₂ marker is filled if Z_{H₂} ≥ 3 on BOTH edges (resolved); else
        drawn as an open circle at 50 % alpha with the failing Z value
        tagged next to it.
        """
        if len(self._comparison_edges) < 2:
            return
        from ..analyze.harmonic import fit_harmonics
        dt = 1.0 / 250.0
        Z_THRESH_H2 = 3.0
        # Source selector: 'sim' uses reconstructed Q(t) (or P(t) for
        # pressure-mode fields) from the stored per-edge harmonics;
        # 'measured' uses PIV Q_t.
        src = getattr(self, 'current_source', 'measured')
        # Pressure mode: when the Quantity selector is P (sim only),
        # the comparison plot reconstructs midpoint P(t) instead of
        # Q(t) and relabels axes accordingly.
        quantity = ('P' if (src == 'sim'
                            and self.current_quantity == 'P')
                    else 'Q')
        # Pick measurement per edge (prefer the loaded-tile measurement
        # in measured mode; sim mode has one synthetic measurement per
        # edge).
        per_edge_data = []
        for (u, v) in self._comparison_edges:
            if not self.G.has_edge(u, v):
                continue
            if src == 'sim':
                m = self._build_sim_measurement(u, v, quantity=quantity)
                if m is None:
                    continue
                per_edge_data.append((u, v, m))
                continue
            meas = self.G.edges[u, v].get('measurements_piv') or []
            if not meas:
                continue
            # Focus-tile rule (same as the colormap + Plot Q(t)):
            # filter > loaded video > best-of-edge fallback.
            tid = (self.current_tile_filter
                    if self.current_tile_filter is not None
                    else self._video_tile_id)
            m = None
            if tid is not None:
                m = next((mm for mm in meas
                          if MeasurementView(mm).tile_id == tid), None)
            if m is None:
                m = _best_measurement(meas)
            # v4 migration: was m.get('Q_t')
            if m is None or MeasurementView(m).Q_t is None:
                continue
            per_edge_data.append((u, v, m))
        if len(per_edge_data) < 2:
            return
        per_edge_data = per_edge_data[:2]

        # ── Regime classification + A/B ordering ───────────────────────
        # Per-edge topo_av_score (0 = arterial side, 1 = venous side,
        # 0.5 = interior).  Pulse-source proximity ordering rule:
        #   arterial-dominated regime (both score < 0.5):
        #     Edge A = smaller graph_dist_art (closer to arterial source)
        #     pulse propagates A → B (same as DC flow)
        #   venous-dominated regime (both score > 0.5):
        #     Edge A = smaller graph_dist_ven (closer to venous sink,
        #              which is the *local* pulse source)
        #     pulse propagates A → B (opposite to DC flow)
        #   cross-regime: ordering is arbitrary; flag in the suptitle
        #              and render row-3 markers as open circles.
        def _edge_topo(u, v):
            d = self.G.edges[u, v]
            return (float(d.get('topo_av_score', np.nan)),
                    float(d.get('graph_dist_art', np.nan)),
                    float(d.get('graph_dist_ven', np.nan)))
        topo = [_edge_topo(u, v) for (u, v, _) in per_edge_data]
        scores = [t[0] for t in topo]
        if all(np.isfinite(scores)) and max(scores) < 0.5:
            regime = 'arterial'
        elif all(np.isfinite(scores)) and min(scores) > 0.5:
            regime = 'venous'
        elif any(not np.isfinite(s) for s in scores):
            regime = 'unknown'
        else:
            regime = 'cross'
        swapped = False
        if regime == 'arterial':
            # smaller graph_dist_art = closer to A source = upstream in pulse-prop
            if topo[1][1] < topo[0][1]:
                per_edge_data = [per_edge_data[1], per_edge_data[0]]
                topo = [topo[1], topo[0]]
                swapped = True
        elif regime == 'venous':
            # smaller graph_dist_ven = closer to V sink (= local pulse source)
            if topo[1][2] < topo[0][2]:
                per_edge_data = [per_edge_data[1], per_edge_data[0]]
                topo = [topo[1], topo[0]]
                swapped = True
        # cross / unknown: leave the user-supplied order
        if swapped:
            # Keep the in-canvas overlays in sync with the final A/B
            # assignment: the user clicked in one order, the regime
            # ordering may have swapped them, so the rings/edge highlight
            # colors must follow suit (A → blue, B → orange).
            new_order = [(u, v) for (u, v, _) in per_edge_data]
            remaining = [e for e in self._comparison_edges
                          if e not in new_order]
            self._comparison_edges = new_order + remaining
            self._refresh_comparison_markers()

        def _wrap(p):
            return (float(p) + np.pi) % (2 * np.pi) - np.pi

        edge_fits = []
        edge_colors = ['C0', 'C1']
        edge_letters = ['A', 'B']
        for (u, v, m) in per_edge_data:
            mv_m = MeasurementView(m)
            # v4 migration: was m['Q_t'] / m.get('f0_hz', 2.5)
            Q_t = np.asarray(mv_m.Q_t, dtype=float)
            T = len(Q_t)
            f0 = float(mv_m.harm.f0_hz or 2.5)
            try:
                hr = fit_harmonics(Q_t, frame_dt=dt, f0=f0, K=3,
                                    loss='huber', include_dc=True)
                a0 = float(hr['a0'])
                harms = {h['k']: h for h in hr['harmonics']}
                h1, h2, h3 = harms.get(1), harms.get(2), harms.get(3)
                phi1 = (float(np.arctan2(-h1['B'], h1['A']))
                        if h1 is not None else float('nan'))
                phi2 = (float(np.arctan2(-h2['B'], h2['A']))
                        if h2 is not None else float('nan'))
                phi3 = (float(np.arctan2(-h3['B'], h3['A']))
                        if h3 is not None else float('nan'))
                amp1 = float(h1['amp']) if h1 is not None else float('nan')
                amp2 = float(h2['amp']) if h2 is not None else float('nan')
                amp3 = float(h3['amp']) if h3 is not None else float('nan')
                resid = np.asarray(hr.get('resid', []), dtype=float)
                sigma = float(np.std(resid)) if resid.size else float('inf')
                se_amp = (sigma * np.sqrt(2.0 / T)
                          if T > 0 else float('inf'))
                z1 = amp1 / max(se_amp, 1e-30) if np.isfinite(amp1) else np.nan
                z2 = amp2 / max(se_amp, 1e-30) if np.isfinite(amp2) else np.nan
                z3 = amp3 / max(se_amp, 1e-30) if np.isfinite(amp3) else np.nan
            except Exception:
                a0 = 0.0
                amp1 = amp2 = amp3 = phi1 = phi2 = phi3 = np.nan
                z1 = z2 = z3 = np.nan
            edge_fits.append(dict(
                u=u, v=v, m=m, Q_t=Q_t, T=T, f0=f0, a0=a0,
                amp1=amp1, amp2=amp2, amp3=amp3,
                phi1=phi1, phi2=phi2, phi3=phi3,
                z1=z1, z2=z2, z3=z3))

        # Quantity-aware axis / title labels for Q vs P modes.
        unit = 'Pa' if quantity == 'P' else 'nL/s'
        sym = quantity  # 'Q' or 'P'
        sym_bar = f'{sym}̄'  # combining macron → Q̄ / P̄
        mean_label = f'mean_{sym}'

        fig = plt.figure(figsize=(11, 11))
        gs = fig.add_gridspec(
            3, 2,
            height_ratios=[1.0, 1.0, 1.0],
            hspace=0.55, wspace=0.30,
            left=0.09, right=0.97, top=0.91, bottom=0.07)

        # ── Top: raw signal, shared y across both panels ──
        ax_raw_A = fig.add_subplot(gs[0, 0])
        ax_raw_B = fig.add_subplot(gs[0, 1], sharey=ax_raw_A)
        for k, ed in enumerate(edge_fits):
            ax = (ax_raw_A, ax_raw_B)[k]
            t_axis = np.arange(ed['T']) * dt
            ax.plot(t_axis, ed['Q_t'], color=edge_colors[k],
                    lw=0.7, alpha=0.9)
            # mean_Q / PI from the fresh harmonic fit (a0, amp1) — canonical
            # graphs don't carry legacy mean_Q/PI keys, so fall back from m.
            a0_k = ed['a0']
            amp1_k = ed['amp1']
            if np.isfinite(a0_k):
                mean_disp = abs(a0_k)
            else:
                # v4 migration: was ed['m'].get('mean_Q')
                _mq = MeasurementView(ed['m']).harm.mean_Q
                mean_disp = float(_mq) if _mq is not None else float('nan')
            if (np.isfinite(amp1_k) and np.isfinite(a0_k)
                    and abs(a0_k) > 0):
                pi_disp = 2.0 * amp1_k / abs(a0_k)
            else:
                # 'PI' is a legacy H1-only Gosling PI; MeasurementView.PI_rms
                # is Parseval-form (all harmonics).  Not exactly the same
                # quantity; keep raw dict-read as best-effort v3 fallback.
                # TODO(v4-migration): PI legacy flat key dropped in v4.
                pi_disp = float(ed['m'].get('PI', float('nan')))
            # SNR_AC = 10·log10(Z1² + Z2² + Z3²) — total cardiac-harmonic
            # power SNR from this edge's fresh fit (DC-free).
            _z_sq_k = sum(z*z for z in (ed['z1'], ed['z2'], ed['z3'])
                           if np.isfinite(z))
            snr_ac_db_k = (10.0 * np.log10(_z_sq_k)
                            if _z_sq_k > 0 else float('nan'))
            ax.set_title(
                f"Edge {edge_letters[k]}: ({ed['u']}, {ed['v']}) "
                # v4 migration: was ed['m'].get('tile_id')
                f"tile {MeasurementView(ed['m']).tile_id}\n"
                f"{mean_label}={mean_disp:.3f}  "
                f"PI={pi_disp:.2f}  "
                f"SNR_AC={snr_ac_db_k:.1f} dB  "
                f"f₀={ed['f0']:.2f} Hz",
                fontsize=9, color=edge_colors[k])
            ax.grid(alpha=0.3)
            ax.set_xlabel('time (s)')
            if k == 0:
                ax.set_ylabel(f'{sym} ({unit})')
        ymins = [np.nanmin(ed['Q_t']) for ed in edge_fits]
        ymaxs = [np.nanmax(ed['Q_t']) for ed in edge_fits]
        ax_raw_A.set_ylim(min(ymins), max(ymaxs))

        # ── Middle: per-edge single-period fold with harmonic decomposition ──
        # Two panels (A on left, B on right), each showing (Q − Q̄) / Q̄
        # over one cycle, decomposed into individual H1, H2, H3 components
        # plus the combined fit.  Mean-normalised so the comparison is of
        # pulsatile fraction shape, not absolute amplitude.
        ax_fold_A = fig.add_subplot(gs[1, 0])
        ax_fold_B = fig.add_subplot(gs[1, 1], sharey=ax_fold_A)
        ax_folds = [ax_fold_A, ax_fold_B]
        N_fold = 200
        phase_grid = np.linspace(0.0, 1.0, N_fold, endpoint=False)
        for k, ed in enumerate(edge_fits):
            ax = ax_folds[k]
            f0 = ed['f0']
            qbar = ed['a0']
            if not (np.isfinite(f0) and f0 > 0
                    and np.isfinite(qbar) and abs(qbar) > 0):
                ax.text(0.5, 0.5, '(no fit)', transform=ax.transAxes,
                         ha='center', va='center', color='#888')
                continue
            color = edge_colors[k]
            T_period = 1.0 / f0
            omega = 2 * np.pi * f0
            t_one = phase_grid * T_period
            qbar_abs = abs(qbar)
            # Individual harmonic components — lighter, dashed/dotted
            harmonic_styles = [
                (1, ed['amp1'], ed['phi1'], '--', 0.7,
                 f"H₁ (|H₁|/{sym_bar}={ed['amp1']/qbar_abs:.2f})"
                 if np.isfinite(ed['amp1']) else None),
                (2, ed['amp2'], ed['phi2'], ':',  0.7,
                 f"H₂ (|H₂|/{sym_bar}={ed['amp2']/qbar_abs:.2f})"
                 if np.isfinite(ed['amp2']) else None),
                (3, ed['amp3'], ed['phi3'], (0, (1, 3)), 0.6,
                 f"H₃ (|H₃|/{sym_bar}={ed['amp3']/qbar_abs:.2f})"
                 if np.isfinite(ed['amp3']) else None),
            ]
            y_total = np.zeros(N_fold)
            for kh, amp_k, phi_k, ls, lw_alpha, lab in harmonic_styles:
                if not (np.isfinite(amp_k) and np.isfinite(phi_k)):
                    continue
                y_k = amp_k * np.cos(kh * omega * t_one + phi_k)
                y_total += y_k
                ax.plot(phase_grid, y_k / qbar_abs,
                         color=color, lw=1.0, alpha=lw_alpha,
                         linestyle=ls, label=lab)
            # Combined (bold)
            ax.plot(phase_grid, y_total / qbar_abs,
                     color=color, lw=2.4, alpha=0.95,
                     label="sum (H₁+H₂+H₃)")
            ax.axhline(0.0, color='#888', lw=0.5, ls=':')
            ax.set_xlim(0, 1)
            ax.set_xlabel('cycle fraction')
            if k == 0:
                ax.set_ylabel(f'({sym} − {sym_bar}) / {sym_bar}')
            # Z-resolution tags
            zs_disp = ' '.join([
                f"Z_H{kh}={ed[f'z{kh}']:.1f}" if np.isfinite(ed[f'z{kh}']) else f"Z_H{kh}=n/a"
                for kh in (1, 2, 3)
            ])
            ax.set_title(
                f"Edge {edge_letters[k]} harmonic decomposition\n"
                f"{zs_disp}",
                fontsize=9, color=edge_colors[k])
            ax.legend(fontsize=7, loc='upper right', framealpha=0.85)
            ax.grid(alpha=0.3)

        # ── Bottom row: per-harmonic transfer function (H₁, H₂) ──
        # Left  — amplitude pulsatile fraction |H_k|/Q̄ for each edge.
        # Right — phase shift Δφ_k between A and B.
        # Both panels overlay a faint √k diffusive prediction so
        # observed vs predicted is a one-glance read.  H₃ is dropped
        # since (a) it's the noisiest and (b) two harmonics already
        # over-determine the diffusive scaling.
        edA, edB = edge_fits[0], edge_fits[1]
        # v4 migration: was edA['m'].get('mean_Q') / edB['m'].get('mean_Q')
        def _fallback_mean_q(_m):
            _mq = MeasurementView(_m).harm.mean_Q
            return float(_mq) if _mq is not None else float('nan')
        qbarA = (edA['a0'] if np.isfinite(edA['a0']) and abs(edA['a0']) > 0
                 else _fallback_mean_q(edA['m']))
        qbarB = (edB['a0'] if np.isfinite(edB['a0']) and abs(edB['a0']) > 0
                 else _fallback_mean_q(edB['m']))
        cross_regime = (regime == 'cross')

        def _pulse_frac(amp, q):
            if not (np.isfinite(amp) and np.isfinite(q)) or q == 0:
                return np.nan
            return amp / abs(q)

        ks = [1, 2]
        amps_A = [edA['amp1'], edA['amp2']]
        amps_B = [edB['amp1'], edB['amp2']]
        phis_A = [edA['phi1'], edA['phi2']]
        phis_B = [edB['phi1'], edB['phi2']]
        zs_A   = [edA.get('z1', np.nan), edA.get('z2', np.nan)]
        zs_B   = [edB.get('z1', np.nan), edB.get('z2', np.nan)]

        pA = [_pulse_frac(amps_A[i], qbarA) for i in range(2)]
        pB = [_pulse_frac(amps_B[i], qbarB) for i in range(2)]

        ax_atten = fig.add_subplot(gs[2, 0])
        ax_phase = fig.add_subplot(gs[2, 1])

        # ── Attenuation panel ───────────────────────────────────────
        # Stems show |H_k|/Q̄ per edge.  Atten annotation in the gap.
        x_off = 0.16
        STEM_LW = 6
        all_p = [p for p in pA + pB if np.isfinite(p) and p > 0]
        for i, k in enumerate(ks):
            zA, zB = zs_A[i], zs_B[i]
            resolved = (np.isfinite(zA) and np.isfinite(zB)
                        and zA >= Z_THRESH_H2 and zB >= Z_THRESH_H2)
            solid = resolved and not cross_regime
            xA, xB = k - x_off, k + x_off
            if np.isfinite(pA[i]) and pA[i] > 0:
                ax_atten.vlines(xA, 1e-4, pA[i], colors='C0',
                                lw=STEM_LW,
                                alpha=0.95 if solid else 0.4)
                ax_atten.plot(xA, pA[i], 'o', ms=12,
                              mfc='C0' if solid else 'none',
                              mec='C0', mew=2.0,
                              alpha=0.95 if solid else 0.7)
            if np.isfinite(pB[i]) and pB[i] > 0:
                ax_atten.vlines(xB, 1e-4, pB[i], colors='C1',
                                lw=STEM_LW,
                                alpha=0.95 if solid else 0.4)
                ax_atten.plot(xB, pB[i], 'o', ms=12,
                              mfc='C1' if solid else 'none',
                              mec='C1', mew=2.0,
                              alpha=0.95 if solid else 0.7)
            # Atten label between the stems.
            if (np.isfinite(pA[i]) and pA[i] > 0
                    and np.isfinite(pB[i]) and pB[i] > 0):
                atten = pB[i] / pA[i]
                ax_atten.annotate(
                    '', xy=(xB, pB[i]), xytext=(xA, pA[i]),
                    arrowprops=dict(arrowstyle='->', color='#444',
                                    lw=1.2,
                                    connectionstyle='arc3,rad=-0.2'))
                ax_atten.text(
                    k, np.sqrt(pA[i] * pB[i]),
                    f"×{atten:.2f}",
                    ha='center', va='center', fontsize=9,
                    color='#222',
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='white', ec='#bbb', alpha=0.9))

        ax_atten.plot([], [], 'o', color='C0', ms=10, label='Edge A')
        ax_atten.plot([], [], 'o', color='C1', ms=10, label='Edge B')
        ax_atten.legend(fontsize=7, loc='upper right')

        ax_atten.set_yscale('log')
        if all_p:
            ax_atten.set_ylim(max(1e-3, 0.3 * min(all_p)),
                              max(all_p) * 1.6)
        else:
            ax_atten.set_ylim(1e-3, 1.0)
        ax_atten.set_xlim(0.5, 2.5)
        ax_atten.set_xticks([1, 2])
        ax_atten.set_xticklabels(['H₁', 'H₂'])
        ax_atten.set_xlabel('harmonic')
        ax_atten.set_ylabel(
            f'|H$_k$| / {sym_bar}  (pulsatile fraction)')
        atten_title = 'Amplitude per harmonic'
        if cross_regime:
            atten_title += ' [cross-regime: hollow]'
        ax_atten.set_title(atten_title, fontsize=10)
        ax_atten.grid(alpha=0.3, which='both', axis='y')

        # ── Phase panel ─────────────────────────────────────────────
        dphis = []
        for i, k in enumerate(ks):
            if np.isfinite(phis_A[i]) and np.isfinite(phis_B[i]):
                dphis.append(np.degrees(_wrap(phis_B[i] - phis_A[i])))
            else:
                dphis.append(np.nan)
        for i, k in enumerate(ks):
            zA, zB = zs_A[i], zs_B[i]
            resolved = (np.isfinite(zA) and np.isfinite(zB)
                        and zA >= Z_THRESH_H2 and zB >= Z_THRESH_H2)
            solid = resolved and not cross_regime
            if np.isfinite(dphis[i]):
                ax_phase.vlines(k, 0, dphis[i],
                                colors='#444', lw=STEM_LW * 0.6,
                                alpha=0.9 if solid else 0.4)
                ax_phase.plot(k, dphis[i], 'o', ms=13,
                              mfc='#444' if solid else 'none',
                              mec='#444', mew=2.0,
                              alpha=0.95 if solid else 0.7)
                ax_phase.text(k + 0.08, dphis[i],
                              f"{dphis[i]:+.0f}°",
                              ha='left', va='center', fontsize=9,
                              color='#222')
        ax_phase.axhline(0.0, color='#888', lw=0.7, ls='--', alpha=0.7)
        # Symmetric y-range around 0 so positive and negative shifts read.
        if any(np.isfinite(d) for d in dphis):
            absmax = max(abs(d) for d in dphis if np.isfinite(d))
            ymax = max(30.0, absmax * 1.25)
            ax_phase.set_ylim(-ymax, ymax)
        else:
            ax_phase.set_ylim(-90, 90)
        ax_phase.set_xlim(0.5, 2.5)
        ax_phase.set_xticks([1, 2])
        ax_phase.set_xticklabels(['H₁', 'H₂'])
        ax_phase.set_xlabel('harmonic')
        ax_phase.set_ylabel('Δφ  (degrees,  B − A)')
        ax_phase.set_title('Phase shift per harmonic', fontsize=10)
        ax_phase.grid(alpha=0.3, axis='y')

        # Title (bold) + subtitle (smaller, italic) for regime context.
        if regime == 'arterial':
            subtitle = ("arterial-dominated pair; pulse propagates "
                        "A → B (DC flow A → B)")
        elif regime == 'venous':
            subtitle = ("venous-dominated pair; pulse propagates A → B "
                        "(DC flow is reversed: B → A)")
        elif regime == 'cross':
            subtitle = ("CROSS-REGIME pair — attenuation does not "
                        "directly apply (markers hollow)")
        else:
            subtitle = ("regime unknown — topo_av_score missing on one "
                        "or both edges")
        main_title = (
            "Edge A vs B — pressure attenuation"
            if quantity == 'P'
            else "Edge A vs B — harmonic attenuation")
        if self._video_tile_id is not None:
            main_title += f"  (tile {self._video_tile_id})"
        fig.suptitle(main_title, fontsize=12, fontweight='bold', y=0.985)
        fig.text(0.5, 0.953, subtitle,
                 ha='center', va='top', fontsize=10, style='italic',
                 color='#444')
        plt.show()


    def _on_plot_boundary_kcl(self):
        """Handler for the 'Plot KCL sum at boundary' button.  Checks
        that the selected edge touches a boundary node, then calls
        `_plot_boundary_kcl_qt` on that boundary side."""
        if self._selected_edge is None:
            return
        u, v = self._selected_edge
        bt_u = self.G.nodes[u].get('boundary_type')
        bt_v = self.G.nodes[v].get('boundary_type')
        if bt_u in ('source', 'sink'):
            boundary_node, junction_node = u, v
        elif bt_v in ('source', 'sink'):
            boundary_node, junction_node = v, u
        else:
            try:
                self._info_box.setPlainText(
                    "Selected edge is not a boundary arm — "
                    "click an edge that touches an A or V boundary node.")
            except Exception:
                pass
            return
        self._plot_boundary_kcl_qt(boundary_node, junction_node)


    def _plot_boundary_kcl_qt(self, boundary_node: int, junction_node: int):
        """For a boundary edge (arm from `boundary_node` at the tip to
        `junction_node` in the network), pop a figure showing the
        KCL-reconstructed Q(t) at the junction (sum of J's OTHER
        incident vessels, sign-flipped so ``+`` = flow out through the
        arm toward the boundary).  This is the "true" boundary Q(t)
        that a downstream sim should use as a BC — NOT the noisy arm
        measurement.

        Overlays the arm's own raw Q(t) traces (per tile) for
        comparison.  Uses tile 14 (the reference tile) for every
        contributing vessel where available; falls back to best-SNR
        per vessel otherwise.

        Prints numeric summary to the info box.
        """
        import matplotlib.pyplot as plt
        REF_TILE = 14

        def _get_meas_at_tile(u, v, tile_id):
            ed = self.G.edges[u, v]
            for m in (ed.get('measurements_piv') or []):
                if m.get('tile_id') == tile_id:
                    return m
            return None

        def _best_meas(u, v):
            ed = self.G.edges[u, v]
            meas = ed.get('measurements_piv') or []
            if not meas:
                return None
            def key(m):
                h = ((m.get('harm') or {}).get('harmonics') or [{}])[0]
                return h.get('snr_db') if h.get('snr_db') is not None else -np.inf
            return max(meas, key=key)

        # Recurse KCL at J for a subgraph excluding the boundary side.
        # For each measured branch use its tile 14 Q_t (or best-SNR fallback).
        # For each unmeasured branch, recurse one hop into the grey node's
        # own neighbors (same rule).
        _visited = {boundary_node, junction_node}
        contributions = []   # list of (edge_repr, tile_id, Q_t signed, mean_Q)
        for nbr in self.G.neighbors(junction_node):
            if nbr in _visited: continue
            m = _get_meas_at_tile(junction_node, nbr, REF_TILE) or _best_meas(junction_node, nbr)
            if m is not None:
                mv = MeasurementView(m)
                if mv.Q_t is None: continue
                Q_t = np.asarray(mv.Q_t, dtype=float)
                # Sign for KCL at J: measurement's flow_from tells us which way flow
                # is defined.  We want "flow OUT of J toward nbr".
                #   If flow_from == J, measurement is J→nbr → sign = +1.
                #   Else (flow_from == nbr) → measurement is nbr→J → sign = -1.
                sign = +1.0 if mv.flow_from == junction_node else -1.0
                contributions.append((f'({junction_node},{nbr})', m.get('tile_id'),
                                      sign * Q_t, sign * (mv.harm.mean_Q or 0)))
            else:
                # Grey branch — one hop recursion into `nbr`'s neighbors
                _visited_inner = _visited | {nbr}
                for nbr2 in self.G.neighbors(nbr):
                    if nbr2 in _visited_inner: continue
                    m2 = _get_meas_at_tile(nbr, nbr2, REF_TILE) or _best_meas(nbr, nbr2)
                    if m2 is None: continue
                    mv2 = MeasurementView(m2)
                    if mv2.Q_t is None: continue
                    Q_t = np.asarray(mv2.Q_t, dtype=float)
                    # At `nbr`, contribution to "flow OUT of nbr toward nbr2".
                    # By KCL at nbr: this equals flow INTO nbr from J side
                    # (= flow OUT of J toward nbr).  Sign per above rule.
                    sign = +1.0 if mv2.flow_from == nbr else -1.0
                    contributions.append(
                        (f'({nbr}→{nbr2}) [via grey]', m2.get('tile_id'),
                         sign * Q_t, sign * (mv2.harm.mean_Q or 0)))

        if not contributions:
            print(f'[boundary Q(t)] No usable contributions at J={junction_node} — cannot reconstruct.')
            return

        # Align lengths (contributions may have different N; take the min)
        min_N = min(len(c[2]) for c in contributions)
        Q_sum = np.zeros(min_N)
        for _, _, Q_t, _ in contributions:
            Q_sum += Q_t[:min_N]
        # KCL sum at J = "sum of flow OUT of J to all OTHER branches" = flow OUT
        # of J toward the boundary arm (via KCL closure).  For a boundary sink,
        # bc_harmonics convention: +DC = flow injected INTO network at boundary.
        # A physical sink drains network → bc DC negative.  Sum here IS the
        # "flow through the arm toward the boundary", so bc = -Q_sum for sinks
        # and bc = +Q_sum for sources.  But for VIEWING the raw KCL sum vs the
        # arm's own Q_t, we plot Q_sum unsigned (as computed).
        mean_sum = float(Q_sum.mean())

        # Also grab the arm's raw traces for comparison
        arm_traces = []
        arm_meas = self.G.edges[boundary_node, junction_node].get(
            'measurements_piv') or []
        for m in sorted(arm_meas,
                         key=lambda mm: (
                             ((mm.get('harm') or {}).get('harmonics') or [{}])[0]
                             .get('snr_db') or -np.inf), reverse=True):
            mv = MeasurementView(m)
            if mv.Q_t is None: continue
            # Same sign rule at boundary_node
            sign = +1.0 if mv.flow_from == boundary_node else -1.0
            arm_traces.append((m.get('tile_id'), sign * np.asarray(mv.Q_t, dtype=float),
                                 (((m.get('harm') or {}).get('harmonics') or [{}])[0])
                                 .get('snr_db')))

        # Build the figure
        fig, ax = plt.subplots(figsize=(11, 5.5))
        t_sum = np.arange(min_N) / 250.0
        # Individual contributions (thin, gray-ish)
        for label, tile_id, Q_t, mq in contributions:
            t = np.arange(min(len(Q_t), min_N)) / 250.0
            ax.plot(t, Q_t[:len(t)], lw=0.7, alpha=0.55,
                     label=f'contrib {label} tile={tile_id}  mean={mq:+.3f}')
        # The KCL sum (bold blue)
        ax.plot(t_sum, Q_sum, lw=2.4, color='C0',
                 label=f'KCL SUM at J={junction_node}  mean={mean_sum:+.3f}')
        # The arm's raw traces (dashed orange, for comparison)
        for tile_id, Q_t, snr in arm_traces:
            t = np.arange(len(Q_t)) / 250.0
            ax.plot(t, Q_t, lw=1.0, ls='--', color='C1', alpha=0.7,
                     label=(f'arm ({boundary_node}→{junction_node}) '
                             f'tile={tile_id}  mean={Q_t.mean():+.3f}  '
                             f'SNR={snr:.1f}dB'))

        boundary_type = self.G.nodes[boundary_node].get('boundary_type', '?')
        ax.set_xlabel('t (s)')
        ax.set_ylabel('Q (nL/s) — signed to match "flow OUT of J toward arm"')
        ax.set_title(f'Boundary Q(t) reconstruction  —  '
                      f'{boundary_type.upper()} {boundary_node},  J={junction_node}\n'
                      f'Bold blue = KCL SUM (what your colleague should use as BC).  '
                      f'Dashed orange = arm\'s own raw Q(t) (untrusted at boundary).')
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        plt.show(block=False)
        try:
            _place_figure_top_right(fig)
        except Exception:
            pass

        # Summary in info box
        lines = [
            f"Boundary Q(t) at {boundary_type.upper()} {boundary_node}",
            f"  J = {junction_node}, contributions = {len(contributions)}",
            f"  KCL SUM  mean_Q = {mean_sum:+.4f} nL/s",
            "  arm raw  mean_Q = " + ', '.join(
                f"{Q_t.mean():+.3f}(tile{tid})" for tid, Q_t, _ in arm_traces),
        ]
        try:
            self._info_box.setPlainText('\n'.join(lines))
        except Exception:
            pass


    def _get_edge_harmonics(self, u, v, source):
        """Return [Z_DC, Z_H1, Z_H2, Z_H3] complex phasors for an edge in
        a given source ∈ {'measured', 'reconciled', 'sim', 'sim_carved'}.
        Returns None if the source has no data for this edge.
        Convention matches compare_waveforms.py: Q(t) = Re(Z0) + Σ Re(Zk·e^{iωkt})."""
        d = self.G.edges[u, v]
        Z = [None, None, None, None]
        if source == 'reconciled':
            a0 = d.get('_h_amp_DC_rec')
            if a0 is not None and np.isfinite(a0):
                Z[0] = complex(float(a0), 0.0)
            for k in (1, 2, 3):
                a = d.get(f'_h_amp_H{k}_rec'); p = d.get(f'_h_phase_H{k}_rec')
                if a is not None and p is not None and np.isfinite(a) and np.isfinite(p):
                    Z[k] = complex(float(a) * np.exp(1j * float(p)))
        elif source == 'sim':
            arr = d.get('_sim_tmp_harmonics')
            if arr is not None:
                arr = np.asarray(arr, dtype=complex)
                for k in range(min(4, len(arr))):
                    Z[k] = complex(arr[k])
        elif source == 'sim_carved':
            a0 = d.get('_h_amp_DC_sim_carved')
            if a0 is not None and np.isfinite(a0):
                Z[0] = complex(float(a0))
            for k in (1, 2, 3):
                a = d.get(f'_h_amp_H{k}_sim_carved')
                p = d.get(f'_h_phase_H{k}_sim_carved')
                if a is not None and p is not None and np.isfinite(a) and np.isfinite(p):
                    Z[k] = complex(float(a) * np.exp(1j * float(p)))
        elif source == 'measured':
            # `_h_Z_DC` / `_h_Z_H{k}` are display-scale proxies (same
            # convention as their _norm counterparts). Use the true
            # amp+phase scalars instead.
            a0 = d.get('_h_amp_DC')
            if a0 is not None and np.isfinite(a0):
                Z[0] = complex(float(a0), 0.0)
            for k in (1, 2, 3):
                a = d.get(f'_h_amp_H{k}'); p = d.get(f'_h_phase_H{k}')
                if a is not None and p is not None and np.isfinite(a) and np.isfinite(p):
                    Z[k] = complex(float(a) * np.exp(1j * float(p)))
            # Last-resort fallback: fit Q_t of the top-quality tile via FFT
            if all(z is None for z in Z):
                meas = d.get('measurements_piv') or []
                if meas:
                    m = sorted(meas, key=lambda mm: (
                        QUALITY_RANK.get(mm.get('quality_tier', 'X'), 0),
                        MeasurementView(mm).snr.h1_db or 0.0), reverse=True)[0]
                    Q_t = m.get('Q_t')
                    if Q_t is not None and len(Q_t) > 8:
                        Qt = np.asarray(Q_t, dtype=float)
                        N = len(Qt)
                        Z[0] = complex(float(np.mean(Qt)), 0.0)
                        # crude one-sided spectrum: pick nearest bin to f0
                        f0 = self._detect_sim_f0_hz()
                        if not np.isfinite(f0): f0 = 2.5
                        dt = float(m.get('frame_dt_s') or m.get('dt_s') or 1.0/250.0)
                        F = np.fft.rfft(Qt - np.mean(Qt)) / (N / 2.0)
                        freqs = np.fft.rfftfreq(N, d=dt)
                        for k in (1, 2, 3):
                            j = int(np.argmin(np.abs(freqs - k*f0)))
                            if 0 < j < len(F):
                                Z[k] = complex(F[j])
        return Z


    def _reconstruct_qt(self, Z, n_cycles=2, n_samples=200):
        """Q(t) = Re(Z_DC) + Σ_{k=1..3} Re(Z_k · exp(i·k·ω₁·t)).
        Returns (t_cycles, Q_arr)."""
        f0 = self._detect_sim_f0_hz()
        if not np.isfinite(f0): f0 = 2.5
        omega1 = 2*np.pi*f0
        t_grid = np.linspace(0.0, n_cycles/f0, n_samples)
        q = np.zeros(n_samples)
        if Z[0] is not None: q += float(np.real(Z[0]))
        for k in (1, 2, 3):
            if Z[k] is None: continue
            q += np.real(Z[k] * np.exp(1j*k*omega1*t_grid))
        return t_grid * f0, q


    def _plot_selected_qt_overlay(self):
        """Overlay Q(t) from the current View-tab source with the
        'Compare with' source from the combo box (both reconstructed
        from stored harmonics DC + H1..H3). Same convention as
        compare_waveforms.py."""
        if self._selected_edge is None:
            return
        u, v = self._selected_edge
        src_a = getattr(self, 'current_source', 'measured')
        src_b = self._overlay_source_combo.currentData()
        if src_b is None or src_b == src_a:
            # nothing to compare — fall back to the ordinary plot
            self._plot_selected_qt()
            return
        Z_a = self._get_edge_harmonics(u, v, src_a)
        Z_b = self._get_edge_harmonics(u, v, src_b)
        # Canonicalise each source so Q̄ ≥ 0 (undirected-edge convention).
        # If DC is negative, flip the ENTIRE phasor set (multiply all
        # harmonics by −1) — that's the same as reversing (u, v) as the
        # reference direction. Preserves waveform shape.
        def _flip_if_neg(Z):
            if Z[0] is None: return Z
            if float(np.real(Z[0])) < 0:
                return [(None if z is None else -z) for z in Z]
            return Z
        Z_a = _flip_if_neg(Z_a)
        Z_b = _flip_if_neg(Z_b)
        if all(z is None for z in Z_a) or all(z is None for z in Z_b):
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.information(self.viewer.window._qt_window, "No overlay",
                f"One of the sources has no phasor data on edge ({u}, {v}).\n"
                f"  {src_a}: {'ok' if any(z is not None for z in Z_a) else 'missing'}\n"
                f"  {src_b}: {'ok' if any(z is not None for z in Z_b) else 'missing'}")
            return
        t_a, q_a = self._reconstruct_qt(Z_a)
        t_b, q_b = self._reconstruct_qt(Z_b)
        import matplotlib
        matplotlib.use('QtAgg', force=False) if hasattr(matplotlib, 'use') else None
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(9, 5), facecolor='white')
        dc_a = float(np.real(Z_a[0])) if Z_a[0] is not None else 0.0
        dc_b = float(np.real(Z_b[0])) if Z_b[0] is not None else 0.0
        h1_a = abs(Z_a[1]) if Z_a[1] is not None else 0.0
        h1_b = abs(Z_b[1]) if Z_b[1] is not None else 0.0
        ax.plot(t_a, q_a, '-',  color='steelblue', lw=1.8,
                 label=f'{src_a}  (Q̄={dc_a:.3f}, |H1|={h1_a:.3f})')
        ax.plot(t_b, q_b, '--', color='crimson',   lw=1.8,
                 label=f'{src_b}  (Q̄={dc_b:.3f}, |H1|={h1_b:.3f})')
        ax.axhline(dc_a, color='steelblue', ls=':', lw=0.7, alpha=0.5)
        ax.axhline(dc_b, color='crimson',   ls=':', lw=0.7, alpha=0.5)
        ax.set_xlabel('cycles'); ax.set_ylabel('Q (per-source native units)')
        ax.set_title(f'Edge ({u}, {v})   {src_a}  vs  {src_b}   '
                      f'|H1|_ratio = {h1_b/max(h1_a,1e-30):.2f}×')
        ax.legend(loc='best'); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.show()


    def _plot_selected_qt(self):
        if self._selected_edge is None:
            return
        # Source-aware:
        #   'sim'        → plot the reconstructed sim waveform
        #   'reconciled' → reconciled waveform (Q\u2032 + aligned harmonics) with the per-tile raw traces
        #   'measured'   → the standard 3-row Q(t) + harmonic decomp + FFT
        src = getattr(self, 'current_source', 'measured')
        if src == 'sim':
            self._plot_selected_sim_qt()
            return
        if src == 'reconciled':
            self._plot_selected_reconciled_qt()
            return
        u, v = self._selected_edge
        meas = self.G.edges[u, v].get('measurements_piv') or []
        if not meas:
            return
        # Rank by quality so colour order is deterministic.
        meas_ranked = sorted(
            meas,
            # v4 migration: mm.get('snr_f0') → MeasurementView(mm).snr.h1_db.
            # quality_tier and snr_pulse are not in the mapping table; keep.
            key=lambda mm: (QUALITY_RANK.get(mm.get('quality_tier', 'X'), 0),
                            (MeasurementView(mm).snr.h1_db or 0),
                            mm.get('snr_pulse', 0) or 0),
            reverse=True)
        # Focus-tile rule: prefer the View-tab tile filter if set
        # (matches what's currently coloured on the network), else the
        # loaded video tile, else fall back to all measurements.
        filter_note = None
        focus_tid = (self.current_tile_filter
                      if self.current_tile_filter is not None
                      else self._video_tile_id)
        if focus_tid is not None:
            filtered = [m for m in meas_ranked
                        if MeasurementView(m).tile_id == focus_tid]
            source = ('tile filter'
                       if self.current_tile_filter is not None
                       else 'currently loaded video')
            if filtered:
                meas_ranked = filtered
                filter_note = (f"showing only tile {focus_tid} "
                                f"({source})")
            else:
                filter_note = (f"this edge has no measurement from "
                                f"tile {focus_tid} ({source}) — "
                                f"showing all instead")

        # Three-row figure:
        #   top = raw Q(t) for every measurement
        #   mid = harmonic decomposition of the best measurement
        #   bot = magnitude spectrum |FFT(Q_t)| (best measurement)
        # The top two share a time axis; the bottom has its own freq axis.
        # 75% of the previous (10×10) inch footprint per user request.
        fig = plt.figure(figsize=(7.5, 7.5))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.85],
                              hspace=0.35)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_spec = fig.add_subplot(gs[2, 0])

        # ── Top: raw Q(t) traces for every measurement ──
        for m in meas_ranked:
            mv_m = MeasurementView(m)
            # v4 migration: was m.get('Q_t')
            Q_t = mv_m.Q_t
            if Q_t is None:
                continue
            Q_t = np.asarray(Q_t, dtype=float)
            # Q is a flow-rate magnitude → display with a positive mean (flip the
            # whole trace if its mean is negative; stored Q_t is already
            # convention-oriented, so this is normally a no-op).
            if np.isfinite(np.nanmean(Q_t)) and np.nanmean(Q_t) < 0:
                Q_t = -Q_t
            t = np.arange(len(Q_t)) / 250.0  # 250 fps
            usable = _measurement_usable(m)
            mark = '✓' if usable else '✗'
            # v4 migration: tile_id / mean_Q via MeasurementView.
            # 'PI' (H1-only Gosling) has no MeasurementView property; use
            # PI_rms (Parseval, all-harmonic) as an approximate substitute.
            # TODO(v4-migration): PI legacy flat key dropped in v4.
            _mq = mv_m.harm.mean_Q
            _mq_disp = float(_mq) if _mq is not None else float('nan')
            _pi_disp = mv_m.PI_rms
            if _pi_disp is None:
                _pi_disp = m.get('PI', float('nan'))
            ax_top.plot(
                t, Q_t,
                lw=0.8 if usable else 0.6,
                ls='-' if usable else '--',
                alpha=0.85 if usable else 0.5,
                label=(f"{mark} tile {mv_m.tile_id}  "
                       f"mean_Q={_mq_disp:.3f}  "
                       f"PI={float(_pi_disp):.2f}"))
        ax_top.set_ylabel('Q (nL/s)')
        title_top = (f'Edge ({u}, {v}) — raw Q(t) across '
                     f'{len(meas_ranked)} of {len(meas)} measurements')
        if filter_note:
            title_top += f'\n({filter_note})'
        ax_top.set_title(title_top)
        ax_top.legend(fontsize=8, loc='best')
        ax_top.grid(alpha=0.3)

        # ── Bottom: harmonic decomposition of the best measurement ──
        # Refit harmonics on the raw Q_t so we get H1 *and* H2 even
        # though the measurement only stores H1 fields directly.
        best = next((m for m in meas_ranked
                     if _measurement_usable(m)
                     and MeasurementView(m).Q_t is not None),
                    None)
        if best is not None:
            from ..analyze.harmonic import fit_harmonics
            best_mv = MeasurementView(best)
            # v4 migration: was best['Q_t'] / best.get('f0_hz', 2.5)
            Q_t = np.asarray(best_mv.Q_t, dtype=float)
            T = len(Q_t)
            t = np.arange(T) / 250.0
            f0 = float(best_mv.harm.f0_hz or 2.5)
            try:
                hr = fit_harmonics(Q_t, frame_dt=1.0/250, f0=f0,
                                    K=3, loss='huber', include_dc=True)
                a0 = float(hr['a0'])
                harms = {h['k']: h for h in hr['harmonics']}
                omega = 2 * np.pi * f0
                # Cumulative reconstructions
                y_dc = np.full(T, a0)
                h1 = harms.get(1)
                y_h1 = (y_dc + h1['A'] * np.cos(omega * t)
                              + h1['B'] * np.sin(omega * t)) if h1 else y_dc
                h2 = harms.get(2)
                y_h2 = (y_h1
                        + h2['A'] * np.cos(2 * omega * t)
                        + h2['B'] * np.sin(2 * omega * t)) if h2 else y_h1
                h3 = harms.get(3)
                y_h3 = (y_h2
                        + h3['A'] * np.cos(3 * omega * t)
                        + h3['B'] * np.sin(3 * omega * t)) if h3 else y_h2
                # Relative phase: phi_k − k·phi_1 (wrapped to [-π, π]).
                # This is what determines waveform shape — when Δφ_rel=0
                # higher harmonics reinforce H1's peak; when ±π they
                # cancel at the peak.
                def _wrap(p):
                    return (float(p) + np.pi) % (2 * np.pi) - np.pi
                # Per-harmonic Z-statistics from the same fit's residuals.
                resid = np.asarray(hr.get('resid', []), dtype=float)
                sigma = float(np.std(resid)) if resid.size else float('inf')
                N = len(Q_t)
                se_dc = sigma / max(np.sqrt(N), 1e-30)
                se_amp = sigma * np.sqrt(2.0 / N) if N > 0 else float('inf')
                z_dc = abs(a0) / max(se_dc, 1e-30)
                z1 = (h1['amp'] / max(se_amp, 1e-30)) if h1 else 0.0
                z2 = (h2['amp'] / max(se_amp, 1e-30)) if h2 else 0.0
                z3 = (h3['amp'] / max(se_amp, 1e-30)) if h3 else 0.0
                snrs = {'DC': z_dc, 'H1': z1, 'H2': z2, 'H3': z3}
                klass = _harmonic_class(snrs)
                # Raw Q(t) shown faintly for noise context; a Butterworth
                # low-pass filtered version (cutoff = 4·f0, just above
                # H3) is overlaid darker so you can see the underlying
                # cardiac shape without the high-frequency junk.
                ax_mid.plot(t, Q_t, color='#bbb', lw=0.5, alpha=0.45,
                            label='raw Q(t)')
                try:
                    from scipy.signal import butter, filtfilt
                    nyq = 0.5 * 250.0   # 125 Hz at 250 fps (was 0.5 / dt with dt undefined: overlay never drawn)
                    cutoff = min(4.0 * f0, 0.9 * nyq)
                    b, a_lp = butter(N=4, Wn=cutoff / nyq, btype='low')
                    Q_lp = filtfilt(b, a_lp, Q_t)
                    ax_mid.plot(t, Q_lp, color='#444', lw=0.9, alpha=0.85,
                                label=f'Q(t) low-passed @ {cutoff:.1f} Hz')
                except Exception:
                    pass
                ax_mid.plot(t, y_dc, color='C2', lw=1.2, ls=':',
                            label=f'DC = {a0:.3f}  (Z={z_dc:.1f})')
                if h1 is not None:
                    ax_mid.plot(t, y_h1, color='C0', lw=1.5,
                                label=f"+ H1 (amp={h1['amp']:.3f}, "
                                      f"PI={2*h1['amp']/max(abs(a0),1e-12):.2f}, "
                                      f"φ={h1['phi']:+.2f}, Z={z1:.1f})")
                if h2 is not None:
                    dphi2 = _wrap(h2['phi'] - 2 * (h1['phi'] if h1 else 0))
                    ax_mid.plot(t, y_h2, color='C3', lw=1.4, ls='--',
                                label=f"+ H2 (amp={h2['amp']:.3f}, "
                                      f"H2/H1={h2['amp']/max(h1['amp'],1e-12):.2f}, "
                                      f"Δφ_rel={dphi2:+.2f}, Z={z2:.1f})")
                if h3 is not None:
                    dphi3 = _wrap(h3['phi'] - 3 * (h1['phi'] if h1 else 0))
                    ax_mid.plot(t, y_h3, color='C1', lw=1.4, ls='-.',
                                label=f"+ H3 (amp={h3['amp']:.3f}, "
                                      f"H3/H1={h3['amp']/max(h1['amp'],1e-12):.2f}, "
                                      f"Δφ_rel={dphi3:+.2f}, Z={z3:.1f})")
                # SNR_AC = 10·log10(Z1² + Z2² + Z3²) — total cardiac-harmonic
                # power SNR, DC-free, on the same dB scale as Q_Hk_snr_db.
                _z_sq = sum(z*z for z in (z1, z2, z3) if np.isfinite(z))
                snr_ac_db = (10.0 * np.log10(_z_sq)
                              if _z_sq > 0 else float('nan'))
                ax_mid.set_title(
                    # v4 migration: was best.get('tile_id')
                    f"Harmonic decomposition — tile {best_mv.tile_id}  "
                    f"(f₀ = {f0:.2f} Hz,  SNR_AC = {snr_ac_db:.1f} dB)")
            except Exception as e:
                ax_mid.text(0.5, 0.5, f'fit_harmonics failed: {e}',
                            ha='center', va='center',
                            transform=ax_mid.transAxes)

            # ── Spectrum panel ──
            # FFT magnitude of the (de-meaned) Q_t to reveal what
            # frequency content is in the signal beyond the fitted
            # harmonics — camera shake, electrical noise, sub-harmonics.
            dt = 1.0 / 250
            Q_dm = Q_t - np.mean(Q_t)
            freqs = np.fft.rfftfreq(T, d=dt)
            mag = (2.0 / T) * np.abs(np.fft.rfft(Q_dm))
            ax_spec.semilogy(freqs, np.maximum(mag, 1e-6),
                              color='#444', lw=0.7)
            # Mark the fitted-harmonic frequencies
            for k in (1, 2, 3):
                ax_spec.axvline(k * f0, color='C0' if k == 1
                                          else 'C3' if k == 2 else 'C1',
                                ls='--', lw=0.9, alpha=0.7,
                                label=f'k·f₀ = {k*f0:.2f} Hz' if k <= 3 else None)
            # Cut off display just past the 4th harmonic — everything
            # higher is noise / aliasing for cardiac-rate signals.
            ax_spec.set_xlim(0, min(4.5 * f0, freqs[-1]))
            ax_spec.set_xlabel('frequency (Hz)')
            ax_spec.set_ylabel('|FFT(Q)|  (nL/s)')
            ax_spec.set_title('Magnitude spectrum  '
                              '(content outside the H1-H3 dashed lines = noise / aliasing)')
            ax_spec.legend(fontsize=8, loc='upper right')
            ax_spec.grid(alpha=0.3, which='both')
        else:
            ax_mid.text(0.5, 0.5,
                        'No usable measurement with Q_t — cannot decompose.',
                        ha='center', va='center',
                        transform=ax_mid.transAxes)
            ax_spec.text(0.5, 0.5, 'No spectrum to show.',
                          ha='center', va='center',
                          transform=ax_spec.transAxes)
        ax_mid.set_xlabel('time (s)'); ax_mid.set_ylabel('Q (nL/s)')
        ax_mid.legend(fontsize=8, loc='best')
        ax_mid.grid(alpha=0.3)
        # Show first (non-blocking) so the window is mapped, THEN
        # resize + move.  Doing it the other way around lets macOS
        # apply its "restore previous window state" rule, which is
        # what was re-maximizing the window despite figsize=(7.5, 7.5).
        plt.show(block=False)
        try:
            _place_figure_top_right(fig)
        except Exception:
            pass
        # With no focus tile (no tile filter, no video), also compare the
        # recordings' waveforms after phase/frequency alignment.
        if focus_tid is None:
            try:
                self._plot_selected_qt_aligned(u, v, meas_ranked)
            except Exception as e:
                print(f"[aligned waveforms] edge ({u}, {v}): {e}")


    def _plot_selected_qt_aligned(self, u, v, meas_ranked, n_bins: int = 40,
                                  peak_at: float = 0.4):
        """Compare every recording of an edge on a common cardiac-phase axis
        (see ``plexusflow.analyze.waveform_shape``).

        (a) folded waveform in nL/s, each recording on its own f0 with H1 peaks matched;
        (b) Q/Q-bar (default), or shape only, (Q - DC)/|H1|, when the Inspect-panel checkbox is ticked;
        (c) shape similarity: upper triangle = noise-corrected correlation r / sqrt(R_a R_b),
            lower triangle = small-sample-corrected F test of the harmonic shape (H2, H3 relative to H1),
            diagonal = split-half reliability R of each recording.
        """
        from ..analyze.waveform_shape import align_recording, compare
        rows = []
        for m in meas_ranked:
            mv = MeasurementView(m)
            if mv.Q_t is None:
                continue
            a = align_recording(mv.Q_t, float(mv.harm.f0_hz or np.nan), n_bins=n_bins, peak_at=peak_at)
            if a is None:
                continue
            a.update(tile=mv.tile_id, usable=_measurement_usable(m))
            rows.append(a)
        if len(rows) < 2:
            return
        import matplotlib.pyplot as plt
        n = len(rows)
        by_mean = bool(getattr(self, '_aligned_norm_mean', True))
        with np.errstate(all='ignore'):
            if by_mean:
                Sd = np.array([r['fold'] / r['a0'] if abs(r['a0']) > 1e-9
                               else np.full(n_bins, np.nan) for r in rows])
            else:
                Sd = np.array([(r['fold'] - r['a0']) / r['amp1'] for r in rows])
            cons = np.nanmedian(Sd, axis=0)
        C = np.full((n, n), np.nan)
        P = np.full((n, n), np.nan)
        for i in range(n):
            C[i, i] = rows[i]['R']
            for j in range(i + 1, n):
                cp = compare(rows[i], rows[j])
                C[i, j] = cp['r_corr']
                C[j, i] = cp['F']
                P[j, i] = cp['p_F']

        fig = plt.figure(figsize=(14, 5.0), facecolor='white')
        gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 0.9], wspace=0.32)
        axA, axB, axC = (fig.add_subplot(gs[0, k]) for k in range(3))
        x = (np.arange(2 * n_bins) + 0.5) / n_bins           # two cycles
        for i, r in enumerate(rows):
            col = f'C{i % 10}'
            ls = '-' if r['usable'] else '--'
            lw = 1.4 if r['usable'] else 0.9
            axA.plot(x, np.tile(r['fold'], 2), color=col, ls=ls, lw=lw,
                     label=(f"{'✓' if r['usable'] else '✗'} tile {r['tile']}  f₀={r['f0']:.2f} Hz  "
                            f"DC={r['a0']:.3f}  ({r['n_cycles']:.0f} cyc)"))
            axB.plot(x, np.tile(Sd[i], 2), color=col, ls=ls, lw=lw,
                     label=(f"tile {r['tile']}  "
                            + (f"PI={2 * r['amp1'] / abs(r['a0']):.2f}  " if by_mean and r['a0'] else '')
                            + f"H2/H1={r['h21']:.2f}  Θ₂₁={r['th21_deg']:+.0f}°  "
                            + f"R={r['R']:.2f}"))
        axB.plot(x, np.tile(cons, 2), color='k', lw=2.4, alpha=0.35, label='median')
        for ax in (axA, axB):
            ax.axvline(1.0, color='0.75', lw=0.6)
            ax.set_xlim(0, 2); ax.grid(alpha=0.3)
            ax.set_xlabel(f'cardiac phase (cycles; H1 peak at {peak_at:g})')
            ax.legend(fontsize=7, loc='upper right', framealpha=0.85)
        axA.axhline(0, color='0.6', lw=0.6)
        axA.set_ylabel('phase-averaged Q (nL/s)')
        axA.set_title('(a) aligned: each recording on its own f₀, H1 peaks matched', fontsize=10)
        if by_mean:
            axB.axhline(1, color='0.6', lw=0.6, ls=':')
            axB.set_ylabel('Q / Q̄')
            axB.set_title('(b) mean-normalised (Q̄ = 1; PI = 2|H1|/Q̄)', fontsize=10)
        else:
            axB.axhline(0, color='0.6', lw=0.6)
            axB.set_ylabel('(Q − DC) / |H1|')
            axB.set_title('(b) shape only (DC removed, H1 amplitude = 1)', fontsize=10)
        # (c) upper: noise-corrected r (colour); lower: chi2/dof (text); diagonal: R
        up = np.where(np.triu(np.ones((n, n), bool), 1), C, np.nan)
        im = axC.imshow(up, vmin=0, vmax=1.2, cmap='viridis')
        labs = [str(r['tile']) for r in rows]
        axC.set_xticks(range(n)); axC.set_xticklabels(labs, fontsize=7, rotation=90)
        axC.set_yticks(range(n)); axC.set_yticklabels(labs, fontsize=7)
        fs = 7 if n <= 8 else 5
        for i in range(n):
            for j in range(n):
                if not np.isfinite(C[i, j]):
                    continue
                if i < j:
                    axC.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center', fontsize=fs,
                             color='k' if C[i, j] > 0.7 else 'w')
                elif i == j:
                    axC.text(j, i, f'R {C[i, j]:.2f}', ha='center', va='center', fontsize=fs, color='0.3')
                else:
                    axC.text(j, i, f'{C[i, j]:.1f}', ha='center', va='center', fontsize=fs,
                             color='#b03030' if P[i, j] < 0.01 else '0.2')
        iu = np.triu_indices(n, 1)
        med_rc = np.nanmedian(C[iu]) if np.isfinite(C[iu]).any() else np.nan
        med_x2 = np.nanmedian(C[(iu[1], iu[0])]) if np.isfinite(C[(iu[1], iu[0])]).any() else np.nan
        axC.set_title('(c) upper: r / √(RₐR_b)  (≈ 1: agree within noise)\n'
                      'lower: shape F of H2, H3 (≈ 1: noise; red: p < 0.01)\n'
                      f'median r_corr = {med_rc:.2f},  median F = {med_x2:.1f}   (r_corr blank if R < 0.3)', fontsize=9)
        fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04, label='noise-corrected r')
        fig.suptitle(f'Edge ({u}, {v}): {n} recordings, phase/frequency aligned '
                     '(sign flipped to positive mean; dashed = not usable; R = split-half reliability)',
                     fontsize=11)
        fig.subplots_adjust(left=0.05, right=0.97, top=0.80, bottom=0.13)
        plt.show(block=False)


    def _plot_selected_reconciled_qt(self):
        """Reconciled waveform of the selected edge — DC Q\u2032 (± ensemble SD) + tile-aligned H1..H3 — over two
        cycles, next to each tile's raw measured mean flow.  Phases are drawn only if the winner tile's time gauge
        is determined; otherwise the waveform shape is shown in that tile's own time origin."""
        if self._selected_edge is None:
            return
        u, v = self._selected_edge
        d = self.G.edges[u, v]
        if d.get('_h_amp_DC_rec') is None:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.information(self.viewer.window._qt_window, "Not reconciled",
                                    f"Edge ({u}, {v}) is not in the carved network (porous region or DA stub).")
            return
        Z = self._get_edge_harmonics(u, v, 'reconciled')
        t, q = self._reconstruct_qt(Z)
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor='white',
                                      gridspec_kw=dict(width_ratios=[2.2, 1]))
        dc, sd = float(d['_h_amp_DC_rec']), float(d.get('_rec_Qp_sd') or 0.0)
        ax.plot(t, q, color='#2ca02c', lw=2.2, label='reconciled')
        ax.axhspan(dc - sd, dc + sd, color='#2ca02c', alpha=0.12, lw=0)
        ax.axhline(dc, color='#2ca02c', ls=':', lw=0.9)
        ax.axhline(0, color='0.6', lw=0.6)
        ac = d.get('_h_amp_H1_rec') is not None
        tag = []
        if d.get('_rec_unstable'):
            tag.append('DC unstable across ensemble')
        if not ac:
            tag.append('no AC field (not eligible)')
        elif not d.get('_rec_phase_ok'):
            tag.append(f"phase not determined (tile {d.get('_rec_tile')} gauge SD {d.get('_rec_gauge_sd', float('nan')):.0f}°)")
        ax.set_title(f"Edge ({u}, {v})   Q\u2032 = {dc:+.3f} ± {sd:.3f} nL/s (u→v)"
                     + (f"   winner tile {d.get('_rec_tile')}" if ac else ''), fontsize=11)
        if tag:
            ax.text(0.02, 0.03, ';  '.join(tag), transform=ax.transAxes, fontsize=9, color='#b03030')
        ax.set_xlabel('cycles'); ax.set_ylabel('Q (nL/s), + = u→v'); ax.grid(alpha=0.3)
        # per-tile raw mean flow, oriented u→v, vs the reconciled value
        rows = []
        for m in d.get('measurements_piv') or []:
            qm = m.get('Q_kl', m.get('mean_Q'))
            if qm is None or not np.isfinite(qm):
                continue
            sgn = 1.0 if (m.get('flow_from'), m.get('flow_to')) != (v, u) else -1.0
            rows.append((int(m.get('tile_id', -1)), sgn * abs(float(qm)) if m.get('flow_from') is not None else float(qm)))
        if rows:
            rows.sort()
            ax2.barh(range(len(rows)), [r[1] for r in rows], color='0.55')
            ax2.set_yticks(range(len(rows))); ax2.set_yticklabels([f"tile {r[0]}" for r in rows], fontsize=8)
        ax2.axvline(dc, color='#2ca02c', lw=2); ax2.axvline(0, color='0.6', lw=0.6)
        ax2.set_xlabel('raw per-tile mean Q (nL/s, u→v)'); ax2.set_title('measurements', fontsize=10)
        fig.tight_layout(); fig.show()


    def _plot_selected_sim_qt(self):
        """Single-edge plot of the SIMULATED waveform (View tab Source =
        Simulated).  Mirrors the measured `_plot_selected_qt` layout but
        reconstructs f(t) from the per-edge complex solver harmonics; plots
        flow Q(t) or midpoint pressure P(t) per the Quantity selector."""
        if self._selected_edge is None:
            return
        u, v = self._selected_edge
        quantity = 'P' if self.current_quantity == 'P' else 'Q'
        m = self._build_sim_measurement(u, v, quantity=quantity)
        if m is None:
            if getattr(self, '_info_box', None) is not None:
                self._info_box.append(
                    "\n[sim] No simulation data on this edge — run the "
                    "Simulate or Pump tab first.")
            return
        harm_key = ('_sim_tmp_p_harmonics' if quantity == 'P'
                    else '_sim_tmp_harmonics')
        arr = np.asarray(self.G.edges[u, v].get(harm_key), dtype=complex)
        unit = 'Pa' if quantity == 'P' else 'nL/s'
        sym = quantity
        sig = np.asarray(m['Q_t'], dtype=float)   # carries P(t) in P-mode
        f0 = float(m['f0_hz'])
        dt = 1.0 / 250.0
        T = len(sig)
        t = np.arange(T) * dt
        omega = 2 * np.pi * f0

        fig = plt.figure(figsize=(7.5, 7.5))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.85],
                              hspace=0.35)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_spec = fig.add_subplot(gs[2, 0])

        # ── Top: reconstructed waveform ──
        ax_top.plot(t, sig, color='C0', lw=1.2,
                    label=(f"sim {sym}(t)   mean={m['mean_Q']:.3g}   "
                           f"PI={m.get('PI', float('nan')):.2f}"))
        ax_top.set_ylabel(f'{sym} ({unit})')
        ax_top.set_title(f'Edge ({u}, {v}) — simulated {sym}(t)   '
                         f'(f₀ = {f0:.2f} Hz)')
        ax_top.legend(fontsize=8, loc='best')
        ax_top.grid(alpha=0.3)

        # ── Mid: exact harmonic decomposition from solver coefficients ──
        def _wrap(p):
            return (float(p) + np.pi) % (2 * np.pi) - np.pi
        colors = {1: 'C0', 2: 'C3', 3: 'C1'}
        styles = {1: '-', 2: '--', 3: '-.'}
        c0 = float(arr[0].real) if len(arr) else 0.0
        amp1 = float(abs(arr[1])) if len(arr) > 1 else 0.0
        phi1 = float(np.angle(arr[1])) if len(arr) > 1 else 0.0
        ax_mid.plot(t, sig, color='#bbb', lw=0.6, alpha=0.5,
                    label='full reconstruction')
        y = np.full(T, c0)
        ax_mid.plot(t, y, color='C2', lw=1.2, ls=':', label=f'DC = {c0:.3g}')
        for k in range(1, min(len(arr), 4)):
            ck = arr[k]
            y = y + (ck.real * np.cos(k * omega * t)
                     - ck.imag * np.sin(k * omega * t))
            ampk = float(abs(ck))
            phik = float(np.angle(ck))
            if k == 1:
                lbl = (f"+ H1 (amp={ampk:.3g}, "
                       f"PI={2*ampk/max(abs(c0),1e-12):.2f}, φ={phik:+.2f})")
            else:
                dphi = _wrap(phik - k * phi1)
                lbl = (f"+ H{k} (amp={ampk:.3g}, "
                       f"H{k}/H1={ampk/max(amp1,1e-12):.2f}, "
                       f"Δφ_rel={dphi:+.2f})")
            ax_mid.plot(t, y, color=colors[k], lw=1.4, ls=styles[k],
                        label=lbl)
        ax_mid.set_xlabel('time (s)')
        ax_mid.set_ylabel(f'{sym} ({unit})')
        ax_mid.set_title('Exact harmonic decomposition '
                         '(solver coefficients — no fit)')
        ax_mid.legend(fontsize=8, loc='best')
        ax_mid.grid(alpha=0.3)

        # ── Bottom: magnitude spectrum (band-limited to H1–H3) ──
        sig_dm = sig - np.mean(sig)
        freqs = np.fft.rfftfreq(T, d=dt)
        mag = (2.0 / T) * np.abs(np.fft.rfft(sig_dm))
        ax_spec.semilogy(freqs, np.maximum(mag, 1e-9), color='#444', lw=0.7)
        for k in (1, 2, 3):
            ax_spec.axvline(k * f0, color=colors[k], ls='--', lw=0.9,
                            alpha=0.7, label=f'k·f₀ = {k*f0:.2f} Hz')
        ax_spec.set_xlim(0, min(4.5 * f0, freqs[-1]))
        ax_spec.set_xlabel('frequency (Hz)')
        ax_spec.set_ylabel(f'|FFT({sym})|  ({unit})')
        ax_spec.set_title('Magnitude spectrum  '
                          '(simulated — content only at k·f₀)')
        ax_spec.legend(fontsize=8, loc='upper right')
        ax_spec.grid(alpha=0.3, which='both')

        plt.show(block=False)
        try:
            _place_figure_top_right(fig)
        except Exception:
            pass
