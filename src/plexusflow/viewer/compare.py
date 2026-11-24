"""Side-by-side comparison of data sources on one colour scale (View tab → Compare).

The network of the current data source stays where it is; every other ticked source (Measured, Reconciled, Model) is
drawn as a copy to the right, over its own copy of the mosaic, with a title above each panel.  All panels — the
original included — are coloured on one scale: Fig 7's fixed scales where they apply (phase lag, relative pulsatile
power), otherwise the colour-bound settings applied to the values of all panels together.  Clicking a vessel in any
panel selects it in all of them; the time series then shows every compared source.  Nothing is written.

Mixin of plexusflow.viewer.app.ReadOnlyMosaicViewer.
"""
from __future__ import annotations

import numpy as np

COMPARE_SOURCES = ('measured', 'reconciled', 'model')
GAP_FRAC = 0.04                       # gap between panels, fraction of the mosaic width
NO_DATA_COLOR = '#555555'


class CompareMixin:

    # ── state ────────────────────────────────────────────────────────
    def _source_available(self, src: str) -> bool:
        if src == 'measured':
            return True
        if src == 'reconciled':
            return self._reconciliation_available()
        if src == 'model':
            return self._model_available()
        return False

    def _source_label(self, src: str) -> str:
        if src == 'measured':
            return "Measured (before projection)" if self._pre_active_for('measured') else "Measured"
        if src == 'reconciled':
            return "Reconciled"
        D = getattr(self, '_model_D', None)
        return f"Model (D = {D * 1e4:.2f}×10⁻⁴ /Pa)" if D else "Model"

    def _pre_active_for(self, src: str) -> bool:
        cur = self.current_source
        try:
            self.current_source = src
            return self._pre_active()
        finally:
            self.current_source = cur

    def _compare_panels(self) -> list:
        """Sources drawn: the current one first (the original network), then the other ticked ones in a fixed order."""
        boxes = getattr(self, '_compare_chks', {})
        ticked = [s for s in COMPARE_SOURCES if s in boxes and boxes[s].isChecked() and self._source_available(s)]
        extra = [s for s in ticked if s != self.current_source]
        return [self.current_source] + extra if extra else []

    def _compare_active(self) -> bool:
        return bool(self._compare_panels()) and not (self.rotate_cw or self.rotate_ccw)

    def _compare_pitch(self) -> float:
        W = float(self.mosaic.shape[1]) if self.mosaic is not None else \
            float(max(self.G.nodes[n].get('x', 0) for n in self.G.nodes()) + 1)
        return W * (1.0 + GAP_FRAC)

    # ── values and colours ───────────────────────────────────────────
    def _values_for(self, src: str) -> list:
        """Per-edge values of the current field for `src` (the viewer's own lookup, without the display filters)."""
        cur = self.current_source
        try:
            self.current_source = src
            if src == 'model' and getattr(self, '_model_D', None) is None and self._model_setup():
                self._model_solve(self._model_D_fit)
            out = []
            for u, v in self.edge_list:
                x = self._resolve_field_value(u, v)
                out.append(float(x) if x is not None and np.isfinite(x) else None)
            return out
        finally:
            self.current_source = cur

    def _shared_style(self, panels):
        """(cmap, norm, transform) common to all panels."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        cur = self.current_source
        styles = []
        for s in panels:
            self.current_source = s; styles.append(self._fig7_style())
        self.current_source = cur
        if styles and all(st is not None for st in styles):
            return styles[0]['cmap'], styles[0]['norm'], (lambda x: x)
        vals = np.array([x for s in panels for x in self._compare_vals[s] if x is not None], float)
        prop = self.current_property
        signed = 'signed' in prop or prop == 'waveform_av'
        use_log = bool(self.log_scale) and not signed and prop != 'phase'
        tf = (lambda x: np.log10(x) if x > 0 else None) if use_log else (lambda x: x)
        eff = np.array([y for y in (tf(x) for x in vals) if y is not None], float) if vals.size else vals
        if eff.size == 0:
            return plt.get_cmap('viridis'), mcolors.Normalize(0, 1), tf
        combo = getattr(self, '_bounds_mode_combo', None)
        if combo is not None and combo.currentText() == 'Manual':
            lo, hi = float(self._vmin_spin.value()), float(self._vmax_spin.value())
        else:
            p_lo = float(self._pct_lo_spin.value()) if getattr(self, '_pct_lo_spin', None) else 2.0
            p_hi = float(self._pct_hi_spin.value()) if getattr(self, '_pct_hi_spin', None) else 98.0
            lo, hi = np.percentile(eff, [min(p_lo, p_hi), max(p_lo, p_hi)])
        if hi <= lo:
            hi = lo + 1e-9
        if signed:
            m = max(abs(lo), abs(hi)); return plt.get_cmap('coolwarm'), mcolors.Normalize(-m, m), tf
        if prop == 'phase':
            return plt.get_cmap('hsv'), mcolors.Normalize(-np.pi, np.pi), tf
        return plt.get_cmap('viridis'), mcolors.Normalize(lo, hi), tf

    def _compare_recolour(self, colors: list) -> list:
        """Called by _refresh_edges before its overlays: colours of the original panel on the shared scale; caches the
        other panels' colours for _compare_draw."""
        import matplotlib.colors as mcolors
        panels = self._compare_panels()
        self._compare_vals = {s: self._values_for(s) for s in panels}
        cmap, norm, tf = self._shared_style(panels)
        self._last_vmin, self._last_vmax = float(norm.vmin), float(norm.vmax)

        def col(x):
            if x is None:
                return NO_DATA_COLOR
            y = tf(x)
            return NO_DATA_COLOR if y is None else mcolors.to_hex(cmap(norm(y)))
        self._compare_cols = {s: [col(x) for x in self._compare_vals[s]] for s in panels}
        return list(self._compare_cols[panels[0]])

    # ── layers ───────────────────────────────────────────────────────
    # ── panel titles: fixed-size labels pinned to the top of the canvas, centred over their panel ──
    def _compare_titles(self, items):
        """items: [(text, world column of the panel centre)].  Plain Qt labels over the canvas: the font size does not
        change with zoom; they stay at the top of the view and follow their panel sideways (hidden when it is off-screen)."""
        from qtpy.QtWidgets import QLabel
        from qtpy.QtCore import Qt
        self._compare_titles_clear()
        canvas = self.viewer.window._qt_viewer.canvas.native
        self._title_labels = []
        for text, xw in items:
            lb = QLabel(text, canvas)
            lb.setStyleSheet("color: white; background: rgba(0, 0, 0, 140); padding: 2px 8px; "
                             "border-radius: 4px; font-size: 13px; font-weight: 600;")
            lb.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            lb.adjustSize(); lb.show(); lb.raise_()
            self._title_labels.append((lb, float(xw)))
        if not getattr(self, '_title_events_connected', False):
            self.viewer.camera.events.center.connect(self._compare_place_titles)
            self.viewer.camera.events.zoom.connect(self._compare_place_titles)
            from qtpy.QtCore import QObject, QEvent

            class _OnResize(QObject):                     # keep titles placed when the canvas is resized
                def __init__(self, cb):
                    super().__init__(); self._cb = cb

                def eventFilter(self, obj, ev):
                    if ev.type() == QEvent.Resize:
                        self._cb()
                    return False
            self._title_resize_filter = _OnResize(self._compare_place_titles)
            canvas.installEventFilter(self._title_resize_filter)
            self._title_events_connected = True
        self._compare_place_titles()

    def _compare_place_titles(self, _event=None):
        labels = getattr(self, '_title_labels', [])
        if not labels:
            return
        canvas = self.viewer.window._qt_viewer.canvas.native
        cw = canvas.width(); cx = float(self.viewer.camera.center[-1]); z = float(self.viewer.camera.zoom)
        for lb, xw in labels:
            x = (xw - cx) * z + cw / 2.0                      # world column → canvas pixel
            lb.setVisible(-lb.width() < x < cw + lb.width())
            lb.move(int(round(x - lb.width() / 2.0)), 10)

    def _compare_titles_clear(self):
        for lb, _ in getattr(self, '_title_labels', []):
            lb.setParent(None); lb.deleteLater()
        self._title_labels = []

    def _compare_clear(self):
        self._compare_titles_clear()
        for lay in getattr(self, '_compare_layers', []):
            try:
                self.viewer.layers.remove(lay)
            except (KeyError, ValueError):
                pass
        self._compare_layers = []

    def _compare_draw(self, lines, widths):
        """Copies of the network (and mosaic) for every panel after the first, and the panel titles."""
        self._compare_clear()
        if not self._compare_active() or not lines:
            return
        panels = self._compare_panels(); pitch = self._compare_pitch()
        base = self.viewer.layers['Mosaic'] if 'Mosaic' in self.viewer.layers else None
        for k, src in enumerate(panels[1:], start=1):
            off = (0.0, k * pitch)
            if base is not None:
                im = self.viewer.add_image(self.mosaic, name=f'Mosaic [{src}]', opacity=base.opacity, translate=off,
                                           contrast_limits=base.contrast_limits, colormap=base.colormap.name)
                self._compare_layers.append(im)
            self._compare_layers.append(self.viewer.add_shapes(
                lines, shape_type='line', edge_color=self._compare_cols[src], edge_width=widths,
                name=f'Edges [{src}]', opacity=0.7, translate=off))
        xs = [p[1] for ln in lines for p in ln]
        xc = 0.5 * (min(xs) + max(xs))
        self._compare_titles([(self._source_label(s), xc + k * pitch) for k, s in enumerate(panels)])
        self._compare_mark_selection()
        if self._edges_layer is not None:                  # keep the original's layers on top for picking / highlights
            self.viewer.layers.selection.active = self._edges_layer

    def _compare_mark_selection(self):
        """Mark the selected vessel in every comparison panel."""
        lay = getattr(self, '_compare_sel_layer', None)
        if lay is not None:
            try:
                self.viewer.layers.remove(lay)
            except (KeyError, ValueError):
                pass
            self._compare_sel_layer = None
        if not self._compare_active() or getattr(self, '_selected_edge', None) is None:
            return
        u, v = self._selected_edge
        if u not in self.G or v not in self.G:
            return
        x = 0.5 * (self.G.nodes[u]['x'] + self.G.nodes[v]['x']); y = self.mosaic_height - 0.5 * (self.G.nodes[u]['y'] + self.G.nodes[v]['y'])
        pitch = self._compare_pitch(); n = len(self._compare_panels())
        pts = np.array([[y, x + k * pitch] for k in range(1, n)])
        try:
            self._compare_sel_layer = self.viewer.add_points(pts, size=40, face_color='transparent',
                                                             border_color='yellow', border_width=0.15, name='Compare selection')
        except TypeError:
            self._compare_sel_layer = self.viewer.add_points(pts, size=40, face_color='transparent',
                                                             edge_color='yellow', name='Compare selection')
        if self._edges_layer is not None:
            self.viewer.layers.selection.active = self._edges_layer

    def _compare_unshift(self, col: float) -> float:
        """Map a click column in any panel back to the original network's column."""
        if not self._compare_active():
            return col
        pitch = self._compare_pitch(); k = int(np.floor(col / pitch))
        return col - k * pitch if 0 < k < len(self._compare_panels()) else col

    # ── controls ─────────────────────────────────────────────────────
    def _build_compare_controls(self):
        from qtpy.QtWidgets import QGroupBox, QHBoxLayout, QVBoxLayout, QCheckBox, QLabel
        box = QGroupBox("Compare side by side (one colour scale)")
        lay = QVBoxLayout(box); row = QHBoxLayout()
        self._compare_chks = {}
        for src, lab in (('measured', 'Measured'), ('reconciled', 'Reconciled'), ('model', 'Model')):
            c = QCheckBox(lab); c.setEnabled(self._source_available(src))
            c.toggled.connect(self._on_compare_toggle); row.addWidget(c); self._compare_chks[src] = c
        row.addStretch(1); lay.addLayout(row)
        note = QLabel("<small>Ticked sources other than the current one are drawn to the right. "
                      "Click a vessel in any panel to select it everywhere.</small>")
        note.setWordWrap(True); lay.addWidget(note)
        return box

    def _on_compare_toggle(self, _checked=None):
        if getattr(self, '_model_box', None) is not None:
            show = self.current_source == 'model' or 'model' in self._compare_panels()
            self._model_box.setVisible(show)
            if show:
                self._enter_model_source()
        self._refresh_edges(); self._refresh_nodes(); self._refresh_cbar()
        try:
            self.viewer.reset_view()
        except Exception:
            pass

    # ── time series ──────────────────────────────────────────────────
    def _plot_selected_compare_qt(self):
        """Q(t) of the selected vessel for every compared source (mean + H1 + H2, two cycles, + = u→v)."""
        if getattr(self, '_selected_edge', None) is None:
            return
        import matplotlib.pyplot as plt
        u, v = self._selected_edge
        fig, ax = plt.subplots(figsize=(8, 4.2), facecolor='white')
        colours = {'measured': '#1f77b4', 'reconciled': '#2ca02c', 'model': '#d62728'}
        for src in self._compare_panels():
            cur = self.current_source
            try:
                self.current_source = src
                Z = self._get_edge_harmonics(u, v, src)
            finally:
                self.current_source = cur
            if Z is None or Z[0] is None:
                continue
            t, q = self._reconstruct_qt([Z[0], Z[1], Z[2], None])
            ax.plot(t, q, color=colours.get(src, 'k'), lw=2.0, ls='--' if src == 'model' else '-', label=self._source_label(src))
        ax.axhline(0, color='0.6', lw=0.6)
        ax.set_xlabel('cycles'); ax.set_ylabel('Q (nL/s), + = u→v'); ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=9)
        ax.set_title(f"Edge ({u}, {v})", fontsize=11)
        fig.tight_layout(); fig.show()
