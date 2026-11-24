"""Selected-vessel views: a live waveform plot docked under the canvas and a compact summary table.

Plot (redrawn on every vessel selection and whenever the Model's distensibility changes), two cardiac cycles on the common
clock, + = the vessel's stored orientation:
  recordings   each tile's two-harmonic fit (synchronised tiles, rotated onto the common clock), thin grey
  Measured     the combined measurement before the projection (1/σ² average of the recordings)
  Reconciled   the conservative field
  Model        the RC network at the current D
Q (nL/s) or Q/Q̄ (each curve over its own mean).  Summary (read-only viewer): geometry, recordings, and for each source the
mean flow (± uncertainty), pulsatility, relative pulsatile power and the H1 / H2 phase lag behind the vitelline veins.

Mixin of plexusflow.viewer.app.ReadOnlyMosaicViewer.  Display only; nothing is written.
"""
from __future__ import annotations

import numpy as np

UM_PER_PX = 1.7
COLOURS = {'recording': '#8a8f98', 'measured': '#4c9be8', 'reconciled': '#4cc38a', 'model': '#ff6b6b'}
LABELS = {'measured': 'Measured', 'reconciled': 'Reconciled', 'model': 'Model'}
SUFFIX = {'measured': '_pre', 'reconciled': '_rec', 'model': '_mod'}


class VesselPanelMixin:

    # ── data ─────────────────────────────────────────────────────────
    def _vessel_harmonics(self, u, v, src):
        """(Q̄, [H1, H2]) of a source for the vessel, in the reconciliation's orientation; None if absent."""
        d = self.G.edges[u, v]; sfx = SUFFIX[src]
        if src == 'model' and self._model_available() and getattr(self, '_model_D', None) is None and self._model_setup():
            self._model_solve(self._model_D_fit)
        q0 = d.get(f'_h_amp_DC{sfx}')
        if q0 is None or not np.isfinite(q0):
            return None
        H = []
        for k in (1, 2):
            a, p = d.get(f'_h_amp_H{k}{sfx}'), d.get(f'_h_phase_H{k}{sfx}')
            H.append(complex(a * np.exp(1j * p)) if a is not None and p is not None and np.isfinite(a) else None)
        return float(q0), H

    def _vessel_recordings(self, u, v):
        """[(tile, Q̄_t, [H1, H2] on the common clock)] for every recording in a synchronised tile, same orientation."""
        A = getattr(self, '_rec_A', None)
        if A is None or not A.get('_meas'):
            return []
        e = tuple(self.G.edges[u, v].get('_rec_uv') or (u, v))
        recs, sgn = A['_meas'].get(e), 1.0
        if recs is None:
            recs, sgn = A['_meas'].get((e[1], e[0])), -1.0
        if not recs:
            return []
        psi = A['psi']; out = []
        for t, r in sorted(recs.items()):
            if t in psi:
                H = np.asarray(r['H'])[:2] * np.exp(-1j * np.arange(1, 3) * psi[t]) * sgn
                out.append((int(t), sgn * float(r['a0']), list(H)))
        return out

    @staticmethod
    def _wave(q0, H, c):
        q = np.full_like(c, q0, dtype=float)
        for k, h in enumerate(H, start=1):
            if h is not None:
                q += np.real(h * np.exp(2j * np.pi * k * c))
        return q

    # ── docked plot ──────────────────────────────────────────────────
    def _build_vessel_dock(self):
        """The waveform plot, docked under the canvas."""
        from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QCheckBox, QLabel
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(4, 2, 4, 2)
        row = QHBoxLayout()
        self._vp_abs = QRadioButton("Q (nL/s)"); self._vp_rel = QRadioButton("Q / Q̄"); self._vp_rel.setChecked(True)
        self._vp_recs = QCheckBox("Recordings"); self._vp_recs.setChecked(True)
        for x in (self._vp_abs, self._vp_rel, self._vp_recs):
            x.toggled.connect(lambda _c: self._vessel_plot_refresh()); row.addWidget(x)
        self._vp_title = QLabel("Click a vessel"); row.addStretch(1); row.addWidget(self._vp_title)
        lay.addLayout(row)
        self._vp_fig = Figure(figsize=(8, 2.4), facecolor='#262930')
        self._vp_canvas = FigureCanvasQTAgg(self._vp_fig)
        self._vp_canvas.setMinimumHeight(200)
        lay.addWidget(self._vp_canvas)
        self.viewer.window.add_dock_widget(w, name="Vessel waveform", area='bottom')
        self._vessel_plot_refresh()

    def _vessel_plot_refresh(self):
        fig = getattr(self, '_vp_fig', None)
        if fig is None:
            return
        fig.clear(); ax = fig.add_axes([0.08, 0.2, 0.70, 0.72]); ax.set_facecolor('#262930')
        for s in ax.spines.values():
            s.set_color('#aaaaaa')
        ax.tick_params(colors='#dddddd', labelsize=8)
        sel = getattr(self, '_selected_edge', None)
        if sel is None or not self.G.has_edge(*sel):
            ax.text(0.5, 0.5, "Click a vessel", color='#dddddd', ha='center', va='center', transform=ax.transAxes)
            self._vp_canvas.draw_idle(); return
        u, v = sel; rel = self._vp_rel.isChecked(); c = np.linspace(0, 2, 400)
        norm = (lambda q0, q: q / q0 if q0 else q * np.nan) if rel else (lambda q0, q: q)
        if self._vp_recs.isChecked():
            recs = self._vessel_recordings(u, v)
            for i, (t, q0, H) in enumerate(recs):
                ax.plot(c, norm(q0, self._wave(q0, H, c)), color=COLOURS['recording'], lw=0.8, alpha=0.7,
                        label=f"recordings ({len(recs)})" if i == 0 else None)
        for src in ('measured', 'reconciled', 'model'):
            got = self._vessel_harmonics(u, v, src)
            if got is None:
                continue
            q0, H = got
            ax.plot(c, norm(q0, self._wave(q0, H, c)), color=COLOURS[src], lw=2.0, ls='--' if src == 'model' else '-',
                    label=LABELS[src] + (f" (D = {self._model_D * 1e4:.2f}×10⁻⁴)" if src == 'model' and getattr(self, '_model_D', None) else ""))
        ax.axhline(1.0 if rel else 0.0, color='#666666', lw=0.6)
        ax.set_xlabel('cardiac cycles (common clock)', color='#dddddd', fontsize=8)
        ax.set_ylabel('Q / Q̄' if rel else 'Q (nL/s), + = u→v', color='#dddddd', fontsize=8)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False, labelcolor='#dddddd')
        self._vp_title.setText(f"Vessel ({u}, {v})")
        self._vp_canvas.draw_idle()

    # ── summary table ────────────────────────────────────────────────
    def _vessel_summary_html(self, u, v) -> str:
        from ..ac.display import phase_lag_deg
        d = self.G.edges[u, v]; ref = self._fig7_phref()
        try:
            from ..reconcile.regions import region_class, LABELS as RL
            cls = region_class(d)
            region = RL.get(cls, cls) if cls else ("network" if '_rec_uv' in d else "outside the reconstructed network")
        except Exception:
            region = "—"
        L, R = d.get('length'), d.get('radius')
        recs = self._vessel_recordings(u, v); n_all = len(d.get('measurements_piv') or [])
        qp = d.get('_h_amp_DC_rec')
        agree = sum(1 for _, q0, _ in recs if qp is not None and np.sign(q0) == np.sign(qp))
        head = (f"<b>Vessel ({u}, {v})</b> — {region}<br>"
                f"length {L * UM_PER_PX:.0f} µm · radius {R * UM_PER_PX:.1f} µm<br>" if L is not None and R is not None else
                f"<b>Vessel ({u}, {v})</b> — {region}<br>")
        head += (f"recordings: {n_all} ({len(recs)} on the common clock"
                 + (f", {agree} in the reconciled direction" if qp is not None and recs else "") + ")")
        cols = [s for s in ('measured', 'reconciled', 'model') if self._vessel_harmonics(u, v, s) is not None]
        if not cols:
            return head + "<br><i>no Measured / Reconciled / Model values here (only the recordings in the plot)</i>"
        unc = {'measured': lambda: abs(d.get('_rec_c') or np.nan) * abs(d.get('_h_amp_DC_pre') or np.nan),
               'reconciled': lambda: (d.get('_rec_Qp_sd') or None), 'model': lambda: None}   # 0 = not estimated

        def fmt(x, f="{:.3g}"):
            return "—" if x is None or not np.isfinite(x) else f.format(x)
        rows = {"Mean flow Q̄ (nL/s)": [], "± uncertainty": [], "Pulsatility 2|H₁|/Q̄": [],
                "Pulsatile / mean power": [], "H₁ lag (°)": [], "H₂ lag (°)": []}
        for s in cols:
            q0, H = self._vessel_harmonics(u, v, s)
            a = [abs(h) if h is not None else None for h in H]
            rows["Mean flow Q̄ (nL/s)"].append(fmt(q0, "{:+.3g}"))
            rows["± uncertainty"].append(fmt(unc[s]()))
            rows["Pulsatility 2|H₁|/Q̄"].append(fmt(2 * a[0] / abs(q0) if a[0] is not None and q0 else None, "{:.2f}"))
            rows["Pulsatile / mean power"].append(fmt((a[0] ** 2 + a[1] ** 2) / q0 ** 2 if None not in a and q0 else None, "{:.2f}"))
            for k in (1, 2):
                lag = phase_lag_deg(H[k - 1], q0, k, ref) if H[k - 1] is not None and ref is not None else None
                rows[f"H{'₁' if k == 1 else '₂'} lag (°)"].append(fmt(lag, "{:.0f}"))
        th = "".join(f"<th style='padding:1px 6px; color:{COLOURS[s]}'>{LABELS[s]}</th>" for s in cols)
        body = "".join(f"<tr><td style='padding:1px 6px'>{k}</td>" + "".join(f"<td align='right' style='padding:1px 6px'>{x}</td>" for x in vals) + "</tr>"
                       for k, vals in rows.items())
        return f"{head}<table style='margin-top:4px'><tr><th></th>{th}</tr>{body}</table>"

    def _on_vessel_selected(self):
        """After a selection: summary table (read-only viewer) and the docked plot."""
        sel = getattr(self, '_selected_edge', None)
        if sel is not None and not self.workbench and self.G.has_edge(*sel):
            self._info_box.setHtml(self._vessel_summary_html(*sel))
        self._vessel_plot_refresh()
