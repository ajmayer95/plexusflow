"""Workbench editing: boundary marking and radial-uncertainty tagging (saved to the stage graph).

Mixin of plexusflow.viewer.app.ReadOnlyMosaicViewer (methods moved verbatim from app.py)."""
from __future__ import annotations

from ._common import (  # noqa: F401
    Path,
    RADIAL_UNCERTAIN_COLOR,
    np,
)


class EditMixin:

    # =================================================================
    # Edit tab: boundary marking + recursive-KCL bc_harmonics extraction.
    # This is the ONE mutation path in the "read-only" viewer — opt-in via
    # the Edit tab's mode selector, otherwise every other tab stays purely
    # visual.  See docs/graph_data_guide.md §2 for the bc_* schema.
    # =================================================================
    def _build_edit_panel(self):
        """Boundary-editing tab.  See class docstring / graph_data_guide §2.

        Mode selector (radio group):
          - Off (default)              — clicks route to the normal
                                          edge-inspection path.
          - Mark source (Shift-click)  — click a node → boundary_type='source'
                                          + recursive junction-KCL bc_harmonics.
          - Mark sink (Ctrl-click)     — click a node → boundary_type='sink'
                                          + recursive junction-KCL bc_harmonics.
          - Mark edge                  — click a node → boundary_type='edge'
                                          (toggle; no bc_harmonics).
          - Clear boundary             — click a node → remove boundary_type
                                          and all bc_* attributes.

        Recursive-KCL parameters:
          - max_depth (default 5)
          - snr_threshold_db (default 3.0)

        Save button writes G back to `self.graph_path` with a rolling
        `.pre_edit_<timestamp>.gpickle` backup.  Auto-save is intentionally
        NOT enabled — you commit when you're ready.
        """
        from qtpy.QtWidgets import (
            QWidget, QVBoxLayout, QGroupBox, QRadioButton, QButtonGroup,
            QSpinBox, QDoubleSpinBox, QGridLayout, QPushButton, QLabel,
            QTextEdit, QCheckBox)
        panel = QWidget()
        lay = QVBoxLayout(panel)

        # Mode selector
        mode_group = QGroupBox("Boundary marking mode")
        mode_lay = QVBoxLayout(mode_group)
        self._edit_mode_btns = QButtonGroup(panel)
        for label, key, tip in [
            ("Off",              None,     "Read-only mode (default)."),
            ("Mark source",     'source', "Click a node → tag as arterial 'source' (boundary flow is emergent from the DC reconciliation)."),
            ("Mark sink",       'sink',   "Click a node → tag as venous 'sink' (boundary flow is emergent from the DC reconciliation)."),
            ("Mark edge",       'edge',   "Click a node → toggle 'edge' tag (no bc_harmonics)."),
            ("Clear at click",  'clear',  "Click a boundary node → remove ITS boundary_type + all its bc_* attrs.  Only the clicked node is affected; other boundaries are untouched."),
            ("Move junction",   'move_junction', "TWO-CLICK: (1) click a graph node that's adjacent to a boundary node (a 'junction'), (2) click a new node.  Re-routes the boundary edge from (old-junction ↔ boundary) to (new-junction ↔ boundary), Use when the current junction is too deep in a low-signal vein — pick a better-signal node closer to the plexus."),
        ]:
            rb = QRadioButton(label)
            rb.setToolTip(tip)
            rb.setChecked(key is None)
            rb.toggled.connect(lambda checked, k=key: self._set_edit_mode(k) if checked else None)
            mode_lay.addWidget(rb)
            self._edit_mode_btns.addButton(rb)
        lay.addWidget(mode_group)


        # Badge-label option: show measured DC flux next to A/V
        badge_group = QGroupBox("Boundary badge labels")
        badge_lay = QVBoxLayout(badge_group)
        self._edit_show_flux_chk = QCheckBox("Show DC flux on A/V badges")
        self._edit_show_flux_chk.setToolTip(
            "When checked, boundary badges show 'A +1.42' / 'V -0.97' "
            "instead of just 'A' / 'V'.  Sign convention: positive = "
            "injected into network (source), negative = removed (sink)."
            "  A sink whose DC came out positive → flow direction "
            "unresolved or mis-tagged.")
        self._edit_show_flux_chk.stateChanged.connect(
            lambda _st: self._refresh_boundary_layers())
        badge_lay.addWidget(self._edit_show_flux_chk)
        lay.addWidget(badge_group)

        # Save
        save_btn = QPushButton("Save graph to disk")
        save_btn.setToolTip(
            "Overwrite the loaded gpickle with the current in-memory graph.  "
            "Writes a .pre_edit_<timestamp>.gpickle backup first.  "
            "Atomic write (temp file + os.replace) so a crash mid-save "
            "can't corrupt the file.")
        save_btn.clicked.connect(self._save_graph_edits)
        lay.addWidget(save_btn)

        # Status log
        lay.addWidget(QLabel("Recent edits:"))
        self._edit_status = QTextEdit(); self._edit_status.setReadOnly(True)
        self._edit_status.setMaximumHeight(240)
        self._edit_status.setPlaceholderText("Recent boundary edits will appear here.")
        lay.addWidget(self._edit_status)

        lay.addStretch()
        return panel


    def _set_edit_mode(self, mode):
        self._edit_mode = mode
        if mode:
            msg = f"[Edit] mode = {mode!r} — click on the mosaic to apply."
        else:
            msg = "[Edit] off — clicks route to normal inspection."
        print(msg)
        if hasattr(self, '_edit_status'):
            self._edit_status.append(msg)


    def _on_edit_click(self, x_graph: float, y_graph: float,
                        max_px: float = 20.0):
        """Handle a click when an Edit-tab mode is active.

        `x_graph`, `y_graph` are in graph coordinates
        (x = G.nodes[n]['x'], y = mosaic_height - row).  Finds the
        nearest node globally (not just existing boundary nodes) and
        applies the current edit mode.
        """
        best_n, best_d = None, float('inf')
        for n in self.G.nodes():
            dx = self.G.nodes[n]['x'] - x_graph
            dy = self.G.nodes[n]['y'] - y_graph
            d = float(np.hypot(dx, dy))
            if d < best_d:
                best_n, best_d = n, d
        if best_n is None or best_d > max_px:
            msg = (f"click ({x_graph:.0f}, {y_graph:.0f}): "
                   f"no node within {max_px:.0f}px")
            self._edit_status.append(msg)
            return
        self._apply_edit_mode(int(best_n), best_d)


    def _apply_edit_mode(self, node: int, click_dist_px: float):
        """Execute the current edit mode's action on `node`."""
        if not self._require_workbench('boundary editing'):
            return
        mode = getattr(self, '_edit_mode', None)
        if mode is None:
            return
        pos = f"(x={self.G.nodes[node]['x']:.0f}, y={self.G.nodes[node]['y']:.0f})"
        if mode == 'clear':
            cur_bt = self.G.nodes[node].get('boundary_type')
            if cur_bt is None:
                self._edit_status.append(
                    f"node {node} {pos}: not a boundary — nothing to clear")
                return
            removed = []
            for k in ('boundary_type', 'bc_harmonics', 'bc_Q_t',
                       'bc_snr_db', 'bc_source', 'bc_n_edges',
                       'bc_junction', 'true_tip', 'max_depth_reached',
                       'n_recursions', 'edges_used'):
                if k in self.G.nodes[node]:
                    del self.G.nodes[node][k]
                    removed.append(k)
            self._edit_status.append(
                f"node {node} {pos}: was {cur_bt!r}, cleared "
                f"{len(removed)} attrs (this node only)")
        elif mode == 'edge':
            cur = self.G.nodes[node].get('boundary_type')
            if cur == 'edge':
                del self.G.nodes[node]['boundary_type']
                self._edit_status.append(f"node {node} {pos}: 'edge' removed")
            else:
                self.G.nodes[node]['boundary_type'] = 'edge'
                self._edit_status.append(f"node {node} {pos}: → edge")
        elif mode in ('source', 'sink'):
            self.G.nodes[node]['boundary_type'] = mode
            self.G.nodes[node]['true_tip'] = True
            # Boundary roles only; boundary flows are emergent from the DC reconciliation
            # (scripts/dc_reconcile.py).  The legacy recursive-KCL bc_harmonics lives in
            # plexusflow.legacy.kcl_reconstruct.
            self._edit_status.append(f"node {node} {pos}: → {mode}")
        # Sync boundary lists + refresh the A/V marker layers so the newly
        # tagged node shows up (or the cleared one disappears) immediately.
        self._refresh_boundary_layers()


    def _refresh_boundary_layers(self):
        """Recompute _source_nodes / _sink_nodes / _boundary_nodes from
        current graph state, then update (or create) the "Arterial
        boundary (A)" and "Venous boundary (V)" Points layers so newly
        tagged nodes are visually badged.

        Handles the S27 case (0 boundary nodes at launch → layers were
        never created) by creating the layers on first mark.  Removes
        layers when a stage drops to zero of that type.
        """
        self._source_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get('boundary_type') == 'source']
        self._sink_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get('boundary_type') == 'sink']
        self._boundary_nodes = self._source_nodes + self._sink_nodes

        def _to_display(n):
            return [self.mosaic_height - self.G.nodes[n]['y'],
                    self.G.nodes[n]['x']]

        show_flux = bool(getattr(self, '_edit_show_flux_chk', None)
                          and self._edit_show_flux_chk.isChecked())

        def _label(n, letter):
            if not show_flux:
                return letter
            bch = self.G.nodes[n].get('bc_harmonics')
            if bch is None or not len(bch):
                return f'{letter} (—)'
            dc = float(np.real(bch[0]))
            return f'{letter} {dc:+.2f}'

        def _sync(existing_attr, node_list, letter, face_color, name):
            layer = getattr(self, existing_attr, None)
            if node_list:
                pts = np.array([_to_display(n) for n in node_list])
                strs = [_label(n, letter) for n in node_list]
                if layer is not None:
                    try:
                        layer.data = pts
                        # napari 0.5+ text is a TextManager; set .string
                        try:
                            layer.text.string = strs
                        except Exception:
                            # older napari: .text is a dict
                            layer.text = {'string': strs, 'size': 12,
                                          'color': 'white', 'anchor': 'center'}
                        layer.visible = True
                    except Exception:
                        import traceback; traceback.print_exc()
                else:
                    try:
                        new_layer = self.viewer.add_points(
                            pts, size=18, face_color=face_color,
                            border_color='white', name=name,
                            text={'string': strs, 'size': 12,
                                  'color': 'white', 'anchor': 'center'})
                    except TypeError:
                        new_layer = self.viewer.add_points(
                            pts, size=18, face_color=face_color,
                            edge_color='white', name=name,
                            text={'string': strs, 'size': 12,
                                  'color': 'white', 'anchor': 'center'})
                    setattr(self, existing_attr, new_layer)
            else:
                # Zero nodes of this type — hide the layer if it exists.
                if layer is not None:
                    try:
                        layer.data = np.empty((0, 2))
                        layer.visible = False
                    except Exception:
                        pass

        _sync('_source_layer', self._source_nodes, 'A',
              '#d62728', 'Arterial boundary (A)')
        _sync('_sink_layer', self._sink_nodes, 'V',
              '#1f77b4', 'Venous boundary (V)')


    def _save_graph_edits(self):
        """Persist Edit-tab changes to disk (atomic, with backup)."""
        if not self._require_workbench('saving edits'):
            return
        from datetime import datetime
        import os, pickle, shutil, tempfile
        target = Path(self.graph_path).resolve()
        if not target.parent.is_dir():
            self._edit_status.append(
                f"save FAILED: parent dir {target.parent} not found")
            return
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = target.with_suffix(f'.pre_edit_{stamp}.gpickle')
        try:
            shutil.copy(str(target), str(backup))
        except Exception as e:
            self._edit_status.append(
                f"save FAILED: could not copy backup: {e}")
            return
        try:
            fd, tmp = tempfile.mkstemp(
                suffix='.edit_save.tmp', dir=str(target.parent),
                prefix='.' + target.stem + '.')
            with os.fdopen(fd, 'wb') as f:
                self._dump_graph(f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, target)
            size_mb = target.stat().st_size / (1024 ** 2)
            self._edit_status.append(
                f"SAVED {target.name} ({size_mb:.0f} MB) "
                f"[backup: {backup.name}]")
        except Exception as e:
            import traceback; traceback.print_exc()
            self._edit_status.append(f"save FAILED: {e}")


    def _toggle_radial_uncertain_on_selected(self):
        """Toggle the selected region tag class on ``self._selected_edge``."""
        if not self._require_workbench('tagging'):
            return
        if self._selected_edge is None:
            self._info_box.setPlainText(
                "No edge selected — click an edge first, then press "
                "'u' or the Toggle button.")
            return
        u, v = self._selected_edge
        if not self.G.has_edge(u, v):
            return
        from ..reconcile.regions import region_class, set_region, LABELS
        cls = getattr(self, '_tag_class', 'topology')
        d = self.G.edges[u, v]
        was = region_class(d) == cls
        set_region(d, None if was else cls)
        self._radial_tags_dirty = True
        action = "cleared" if was else "tagged"
        self._info_box.setPlainText(
            f"Edge ({u}, {v}) {action}: {LABELS[cls]}.")
        self._refresh_edges()
        self._refresh_radial_status()


    def _refresh_radial_status(self):
        """Update the status label with the count of tagged edges."""
        if not hasattr(self, '_radial_status_lbl'):
            return
        from ..reconcile.regions import counts
        c = counts(self.G)
        dirty = " (unsaved)" if self._radial_tags_dirty else ""
        self._radial_status_lbl.setText(
            f"Tagged: {c['topology']} topology-uncertain, {c['open']} open-boundary{dirty}")


    def _on_click_to_tag_toggle(self, checked: bool):
        """Enter/exit click-to-tag mode."""
        self._click_to_tag_mode = bool(checked)
        if checked:
            self._info_box.setPlainText(
                "Click-to-tag mode ON. Every click on a vessel toggles the "
                "selected tag class. Uncheck to return to normal.")
        else:
            self._info_box.setPlainText(
                "Click-to-tag mode OFF.")


    def _tag_nearest_edge(self, x_graph: float, y_graph: float,
                            max_px: float = 20.0):
        """Toggle the selected tag class on the edge nearest ``(x, y)`` in
        graph coordinates.  Used by click-to-tag mode + rectangle tag."""
        import numpy as np
        if self._edge_midpoints is None or len(self._edge_midpoints) == 0:
            return
        d2 = ((self._edge_midpoints[:, 0] - x_graph) ** 2 +
              (self._edge_midpoints[:, 1] - y_graph) ** 2)
        i = int(np.argmin(d2))
        dist = float(np.sqrt(d2[i]))
        if dist > max_px:
            return
        u, v = self.edge_list[i]
        from ..reconcile.regions import region_class, set_region
        cls = getattr(self, '_tag_class', 'topology')
        d = self.G.edges[u, v]
        set_region(d, None if region_class(d) == cls else cls)
        self._radial_tags_dirty = True
        self._refresh_edges()
        self._refresh_radial_status()


    def _ensure_region_layer(self):
        """Create the scratch Shapes layer used for region tag/untag if
        it doesn't exist yet, and register the mouse-drag callback ONCE.
        Reusing a single layer across button-presses avoids the napari
        add/remove race that made the old rectangle tag stop working
        after a couple of uses."""
        lyr = getattr(self, '_region_tag_layer', None)
        # `lyr not in self.viewer.layers` covers the case where the user
        # deleted the layer manually via napari's layer panel.
        if lyr is not None and lyr in self.viewer.layers:
            return lyr
        lyr = self.viewer.add_shapes(
            [], name='RegionTag',
            edge_color=__import__('plexusflow.reconcile.regions', fromlist=['COLORS']).COLORS.get(getattr(self, '_tag_class', 'topology'), RADIAL_UNCERTAIN_COLOR),
            face_color='transparent', edge_width=2.0, opacity=0.9,
        )
        self._region_tag_layer = lyr

        # Act on each shape only once napari has fully committed it
        # (data event, action 'added'), deferred to the next event-loop
        # tick.  The old mouse-drag callback fired on the FIRST release of
        # a click-trace-click lasso, cleared the layer while napari was
        # still building the shape, and crashed napari's lasso handler.
        from qtpy.QtCore import QTimer

        def _on_region_data(event):
            if str(getattr(event, 'action', '')) == 'added':
                QTimer.singleShot(0, self._on_region_shape_added)

        lyr.events.data.connect(_on_region_data)
        return lyr


    def _on_region_shape_added(self):
        """Apply the pending tag/untag to the shape just drawn (rectangle,
        ellipse, lasso or polygon), clear it, and stay armed for the next."""
        import numpy as np
        lyr = getattr(self, '_region_tag_layer', None)
        if lyr is None or lyr not in self.viewer.layers or not len(lyr.data):
            return
        poly_rc = np.asarray(lyr.data[-1], dtype=float)[:, -2:]
        if lyr.shape_type[-1] == 'ellipse' and len(poly_rc) == 4:
            c = poly_rc.mean(axis=0)
            a = 0.5 * (poly_rc[1] + poly_rc[2]) - c
            b = 0.5 * (poly_rc[2] + poly_rc[3]) - c
            t = np.linspace(0, 2 * np.pi, 96, endpoint=False)[:, None]
            poly_rc = c + np.cos(t) * a + np.sin(t) * b
        action = getattr(self, '_region_action_pending', 'tag')
        if len(poly_rc) >= 3:
            n = self._apply_region_action(poly_rc, action)
            if n:
                self._radial_tags_dirty = True
            verb = "Tagged" if action == "tag" else "Untagged"
            self._info_box.setPlainText(
                f"{verb} {n} vessel(s) via region.  Draw another region to "
                f"{action} more, or press Stop.")
            self._refresh_edges()
            self._refresh_radial_status()
        try:
            mode = lyr.mode
            lyr.data = []
            idx = self.viewer.layers.index(lyr)
            if idx != len(self.viewer.layers) - 1:
                self.viewer.layers.move(idx, len(self.viewer.layers) - 1)
            self.viewer.layers.selection.active = lyr
            lyr.mode = mode
        except Exception:
            pass


    def _reset_region_layer(self):
        """Clear scratch shapes and drop out of draw mode so accidental
        clicks don't add more shapes.  Layer itself is kept for reuse."""
        lyr = getattr(self, '_region_tag_layer', None)
        if lyr is None or lyr not in self.viewer.layers:
            return
        try:
            lyr.data = []
            lyr.mode = 'pan_zoom'
        except Exception:
            pass


    def _start_region_action(self, action: str):
        """action ∈ {'tag', 'untag'}.  Sets the scratch shapes layer's
        mode to the currently-selected shape (Rectangle / Lasso) and
        records the pending action so the drag callback knows what to
        apply on release.

        Robustness: the scratch layer is moved to the TOP of the layer
        stack every time we arm, and Qt events are pumped between the
        active-layer change and the mode change.  Without this, after a
        few sessions napari's mouse-event dispatcher was ending up on the
        Edges or Mosaic layer (whichever was frontmost) and reading the
        drag as a pan/zoom instead of a shape-add."""
        if not self._require_workbench('region tagging'):
            return
        from qtpy.QtWidgets import QApplication
        shape_mode = self._region_shape_combo.currentData()   # 'add_rectangle' / 'add_polygon_lasso'
        self._region_action_pending = action
        self._region_shape_pending = shape_mode
        lyr = self._ensure_region_layer()
        try:
            lyr.data = []
        except Exception:
            pass
        # Force the scratch layer to be visible, top of the stack, and
        # the interactive-active layer — belt and suspenders against
        # napari's mouse-dispatch quirks.
        try:
            lyr.visible = True
            idx = self.viewer.layers.index(lyr)
            top = len(self.viewer.layers) - 1
            if idx != top:
                self.viewer.layers.move(idx, top)
        except Exception:
            pass
        self.viewer.layers.selection.active = lyr
        # Let napari process the active-layer change before we set the
        # mode; without this pump the mode change occasionally lands on
        # a stale layer state and the next drag reads as pan/zoom.
        try:
            QApplication.processEvents()
        except Exception:
            pass
        lyr.mode = shape_mode
        shape_label = self._region_shape_combo.currentText()
        self._info_box.setPlainText(
            f"{shape_label} {action}: draw regions on the canvas — each "
            f"is applied when the shape closes.  Every edge whose midpoint "
            f"is inside gets the selected tag class "
            f"{'set' if action == 'tag' else 'cleared'}.  Press Stop when done.")


    def _apply_region_action(self, poly_rc, action: str):
        """poly_rc: (N, 2) polygon vertices in display (row, col) coords.
        action ∈ {'tag', 'untag'}.  Returns the count of edges whose flag
        actually changed (i.e. was toggled, not already in target state)."""
        import numpy as np
        from matplotlib.path import Path
        H = self.mosaic_height
        # edge midpoints stored as (x, y) with y in graph frame → display
        # (row, col) = (H − y, x)
        disp = np.column_stack([H - self._edge_midpoints[:, 1],
                                 self._edge_midpoints[:, 0]])
        inside = Path(np.asarray(poly_rc)).contains_points(disp)
        from ..reconcile.regions import region_class, set_region
        cls = getattr(self, '_tag_class', 'topology')
        n = 0
        for i in np.flatnonzero(inside):
            u, v = self.edge_list[i]
            d = self.G.edges[u, v]
            cur = region_class(d)
            if action == 'tag' and cur != cls:
                set_region(d, cls)
                n += 1
            elif action == 'untag' and cur == cls:
                set_region(d, None)
                n += 1
        return n


    def _save_radial_uncertain_tags(self):
        """Write the current graph back to disk with updated flags."""
        if not self._require_workbench('saving tags'):
            return
        import pickle
        from qtpy.QtWidgets import QMessageBox
        try:
            with open(self.graph_path, 'wb') as f:
                self._dump_graph(f)
            self._radial_tags_dirty = False
            self._refresh_radial_status()
            from ..reconcile.regions import counts
            _c = counts(self.G); n = f"{_c['topology']} topology-uncertain + {_c['open']} open-boundary"
            QMessageBox.information(
                self.viewer.window._qt_window,
                "Region tags saved",
                f"Wrote {n} tagged vessel(s) to:\n{self.graph_path}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self.viewer.window._qt_window,
                "Save failed",
                f"Could not save graph:\n{exc}",
            )
