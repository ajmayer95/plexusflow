"""'Model' data source: the distributed RC network of Fig 7 at a user-chosen wall distensibility D (flow and pressure).

The model is driven exactly as in scripts/make_figure7.py: mean flow = the passive network flow with the reconciliation's
boundary fluxes (dc_reconciled_v3 `Qs`); harmonics 1–2 from plexusmech.ac.model.solve with the boundary forcing stored by
the distensibility fit (results/ac_estimate_v11.json: reconciled boundary flux × measured waveform, veins sharing one
waveform).  Only D changes.  Pressure (Quantity P): vessel mean = mean of its end nodes, drop = tail − head, in Pa; the mean
pressure is gauged to 0 at a vein node.  Results live in memory as `_h_*_mod` edge attributes and are never written.

Mixin of plexusflow.viewer.app.ReadOnlyMosaicViewer; needs `plexusmech` and the reconciliation results.
"""
from __future__ import annotations
import json

import numpy as np

MODEL_FIT = ('ac_estimate_v11.json', 'ac_estimate_v10.json')   # distensibility fit the model forcing and D̂ are read from (Fig 7's; v10: the earlier loss)
D_RANGE = (1.0e-4, 1.5e-3)                  # slider range, Pa⁻¹ (log scale)
N_STEPS = 200


class ModelMixin:

    def _model_available(self) -> bool:
        return bool(getattr(self, 'mech_available', False)) and bool(getattr(self, '_rec', None)) \
            and getattr(self, '_rec_A', None) is not None and self._model_fit_path() is not None

    def _model_fit_path(self):
        res = self._results_dir()
        return next((res / f for f in MODEL_FIT if (res / f).exists()), None) if res is not None else None

    def _model_setup(self) -> bool:
        """Build the solver context once (Fig 7 recipe).  Returns False if the model cannot be built."""
        if getattr(self, '_model_ctx', None) is not None:
            return True
        if not self._model_available():
            return False
        from ..reconcile import carved_domain
        from plexusmech.ac.model import build_context
        D, A = self._rec_D, self._rec_A
        J = json.loads(self._model_fit_path().read_text())
        dom = carved_domain(self.G)
        if [tuple(e) for e in dom['edges']] != [tuple(e) for e in D['edges']]:
            print("  model: carved network differs from the reconciliation's — Model source disabled")
            return False
        C = build_context(D, A, dom)
        nix = {n: i for i, n in enumerate(dom['nodes'])}
        C['F'] = dict(C['F'])
        for k in (1, 2):                                     # boundary forcing as stored by the fit (as Fig 7)
            F = np.zeros(C['nN'], complex)
            F[[nix[n] for n in J['nodes']]] = [complex(a, b) for a, b in J[f'f{k}']]
            C['F'][(k, 'flux')] = F
        self._model_ctx = C
        self._model_edges = [tuple(e) for e in dom['edges']]
        self._model_D_fit = float(J.get('D_parabola', J['D']))
        self._model_D = None
        from plexusmech.ac.model import dc_node_pressure
        P0 = dc_node_pressure(C, D['fb']); t, h = C['tail'], C['head']
        for i, (u, v) in enumerate(self._model_edges):
            d = self.G.edges[u, v]
            d['_h_amp_DC_mod'] = float(C['Qs_dc'][i])
            d['_h_p_DC_mod'] = float(0.5 * (P0[t[i]] + P0[h[i]])); d['_h_dp_DC_mod'] = float(P0[t[i]] - P0[h[i]])
        print(f"  model: RC network on {len(self._model_edges)} carved edges, forcing from {self._model_fit_path().name} "
              f"(fitted D̂ = {self._model_D_fit:.2e} /Pa)")
        return True

    def _model_solve(self, D: float) -> None:
        """Harmonics 1–2 at distensibility D → `_h_amp_H{k}_mod`, `_h_phase_H{k}_mod` (in memory)."""
        if not self._model_setup():
            return
        from plexusmech.ac.model import node_pressure, NL
        C = self._model_ctx; t, h = C['tail'], C['head']
        for k in (1, 2):
            P, Ys, Yt = node_pressure(C, k, float(D), 0.0, 'flux')     # purely elastic wall, as the fit
            H = 0.5 * (Ys + Yt) * (P[t] - P[h]) / NL          # vessel-mean flux, as plexusmech.ac.model.solve
            Pe, dP = 0.5 * (P[t] + P[h]), P[t] - P[h]
            for i, (u, v) in enumerate(self._model_edges):
                d = self.G.edges[u, v]
                d[f'_h_amp_H{k}_mod'] = float(abs(H[i])); d[f'_h_phase_H{k}_mod'] = float(np.angle(H[i]))
                d[f'_h_p_H{k}_mod'] = complex(Pe[i]); d[f'_h_dp_H{k}_mod'] = complex(dP[i])
        self._model_D = float(D)

    # ── controls (View tab) ──────────────────────────────────────────
    def _build_model_controls(self):
        """Distensibility slider (log scale) + fitted-D̂ reset; shown only while the Model source is selected."""
        from qtpy.QtWidgets import QGroupBox, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QPushButton
        from qtpy.QtCore import Qt
        box = QGroupBox("Model: wall distensibility D")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        self._model_slider = QSlider(Qt.Horizontal)
        self._model_slider.setRange(0, N_STEPS)
        self._model_label = QLabel("")
        self._model_label.setMinimumWidth(130)
        row.addWidget(self._model_slider, 1); row.addWidget(self._model_label)
        lay.addLayout(row)
        reset = QPushButton("Reset to fitted D̂")
        lay.addWidget(reset)
        note = QLabel("<small>RC network driven by the reconciled boundary fluxes (Fig 7). "
                      "Temporary: recomputed on the fly, never saved.</small>")
        note.setWordWrap(True)
        lay.addWidget(note)
        self._model_slider.valueChanged.connect(self._on_model_slider)
        reset.clicked.connect(lambda: self._set_model_D(getattr(self, '_model_D_fit', 4e-4)))
        box.setVisible(False)
        self._model_box = box
        return box

    def _slider_to_D(self, s: int) -> float:
        lo, hi = np.log10(D_RANGE[0]), np.log10(D_RANGE[1])
        return float(10 ** (lo + (hi - lo) * s / N_STEPS))

    def _D_to_slider(self, D: float) -> int:
        lo, hi = np.log10(D_RANGE[0]), np.log10(D_RANGE[1])
        return int(round(np.clip((np.log10(D) - lo) / (hi - lo), 0, 1) * N_STEPS))

    def _set_model_D(self, D: float) -> None:
        self._model_slider.setValue(self._D_to_slider(D))     # → _on_model_slider
        self._on_model_slider(self._model_slider.value(), exact=D)

    def _on_model_slider(self, s: int, exact=None) -> None:
        D = float(exact) if exact is not None else self._slider_to_D(s)
        fit = getattr(self, '_model_D_fit', None)
        tag = "  (fitted)" if fit is not None and abs(np.log10(D / fit)) < 0.5 * np.log10(D_RANGE[1] / D_RANGE[0]) / N_STEPS else ""
        self._model_label.setText(f"D = {D * 1e4:.2f}×10⁻⁴ /Pa{tag}")
        if getattr(self, 'current_source', None) != 'model' and 'model' not in self._compare_panels():
            return
        self._model_solve(D)
        self._refresh_edges(); self._refresh_nodes(); self._refresh_cbar()
        if hasattr(self, '_vessel_plot_refresh'):
            self._vessel_plot_refresh()
            sel = getattr(self, '_selected_edge', None)
            if sel is not None and not self.workbench and self.G.has_edge(*sel):
                self._info_box.setHtml(self._vessel_summary_html(*sel))

    def _enter_model_source(self) -> None:
        """Called when the Model source is selected: build once, solve at the current (initially fitted) D."""
        if not self._model_setup():
            return
        if self._model_D is None or not self._model_label.text():
            self._model_slider.blockSignals(True)
            self._model_slider.setValue(self._D_to_slider(self._model_D_fit))
            self._model_slider.blockSignals(False)
            self._on_model_slider(self._model_slider.value(), exact=self._model_D_fit)
