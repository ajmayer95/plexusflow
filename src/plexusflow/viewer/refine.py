"""Refine tab (workbench): iterative local refinement of the graph.

    1 Region      draw a polygon (lasso / polygon) on the mosaic
    2 Mask        optional: paint / erase the vessel mask (only edits INSIDE the region are used)
    3 Re-vectorise   net3 on the region, spliced into the graph → preview (red = replaced, green = new, yellow = dead end)
    4 Measure + refit   background job (project venv): measures only the new vessels, refits current vs candidate
    5 Accept / Discard  accept writes the graph (and mask) through the patch log (graph/patches); discard deletes it
    Patch log     list of accepted patches; revert the selected one

Library: plexusflow.edit.splice / refine / patches.  Nothing here writes the stage until Accept or Revert.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REFINE_REGION = "Refine region"
REFINE_PREVIEW = "Refine preview"
REFINE_MASK = "Refine mask"


def _job_python() -> str:
    """Python for background jobs: $PLEXUSFLOW_JOB_PYTHON, else the repo's .venv, else this interpreter."""
    env = os.environ.get("PLEXUSFLOW_JOB_PYTHON")
    if env:
        return env
    repo = Path(__file__).resolve().parents[3]
    for cand in (repo / ".venv" / "bin" / "python", repo / ".venv" / "Scripts" / "python.exe"):
        if cand.exists():
            return str(cand)
    return sys.executable


class RefineMixin:
    """Mixed into ReadOnlyMosaicViewer (workbench only)."""

    # ── panel ──────────────────────────────────────────────────────────────────────────────────────────────────────
    def _build_refine_panel(self):
        from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QComboBox,
                                    QPlainTextEdit, QLineEdit, QListWidget, QCheckBox)
        w = QWidget(); lay = QVBoxLayout(w)
        self._rf = dict(poly=None, result=None, pending=None, proc=None, mask_layer=None)

        intro = QLabel("<b>Local refinement</b>: draw a region, optionally fix the mask there, re-vectorise it, "
                       "measure the new vessels and compare the DC fit before accepting.  Every accepted change is a "
                       "reversible patch (graph/patches).")
        intro.setWordWrap(True); lay.addWidget(intro)

        # 1 region
        g1 = QGroupBox("1  Region"); l1 = QVBoxLayout(g1)
        row = QHBoxLayout(); self._rf_shape = QComboBox()
        for lab, mode in (("Lasso", "add_polygon_lasso"), ("Polygon", "add_polygon"), ("Rectangle", "add_rectangle")):
            self._rf_shape.addItem(lab, userData=mode)
        b_draw = QPushButton("Draw region"); b_draw.clicked.connect(self._rf_start_draw)
        b_clear = QPushButton("Clear"); b_clear.clicked.connect(self._rf_clear)
        row.addWidget(self._rf_shape); row.addWidget(b_draw); row.addWidget(b_clear); l1.addLayout(row)
        self._rf_region_lbl = QLabel("no region"); l1.addWidget(self._rf_region_lbl)
        lay.addWidget(g1)

        # 2 mask
        g2 = QGroupBox("2  Mask (optional)"); l2 = QVBoxLayout(g2)
        row = QHBoxLayout()
        b_m = QPushButton("Edit mask"); b_m.setToolTip("Adds an editable 'Refine mask' layer (paint = vessel, "
                                                       "erase = background).  Only edits inside the region are used.")
        b_m.clicked.connect(self._rf_edit_mask)
        b_mr = QPushButton("Reset mask edits"); b_mr.clicked.connect(self._rf_reset_mask)
        row.addWidget(b_m); row.addWidget(b_mr); l2.addLayout(row)
        self._rf_mask_lbl = QLabel("mask: stage mask (unedited)"); l2.addWidget(self._rf_mask_lbl)
        lay.addWidget(g2)

        # 3 re-vectorise
        g3 = QGroupBox("3  Re-vectorise region"); l3 = QVBoxLayout(g3)
        b_v = QPushButton("Re-vectorise + preview"); b_v.clicked.connect(self._rf_splice); l3.addWidget(b_v)
        lay.addWidget(g3)

        # 4 measure + refit
        g4 = QGroupBox("4  Measure + refit"); l4 = QVBoxLayout(g4)
        row = QHBoxLayout(); self._rf_measure_chk = QCheckBox("measure new vessels"); self._rf_measure_chk.setChecked(True)
        self._rf_eval_btn = QPushButton("Run (background)"); self._rf_eval_btn.clicked.connect(self._rf_evaluate)
        self._rf_stop_btn = QPushButton("Stop"); self._rf_stop_btn.clicked.connect(self._rf_stop); self._rf_stop_btn.setEnabled(False)
        row.addWidget(self._rf_measure_chk); row.addWidget(self._rf_eval_btn); row.addWidget(self._rf_stop_btn); l4.addLayout(row)
        lay.addWidget(g4)

        # 5 accept
        g5 = QGroupBox("5  Accept / discard"); l5 = QVBoxLayout(g5)
        self._rf_note = QLineEdit(); self._rf_note.setPlaceholderText("reason (from the imaging), e.g. 'marginal band re-segmented'")
        l5.addWidget(self._rf_note)
        row = QHBoxLayout()
        self._rf_accept_btn = QPushButton("Accept → stage"); self._rf_accept_btn.clicked.connect(self._rf_accept)
        b_dis = QPushButton("Discard"); b_dis.clicked.connect(self._rf_discard)
        row.addWidget(self._rf_accept_btn); row.addWidget(b_dis); l5.addLayout(row)
        lay.addWidget(g5)

        # output
        self._rf_out = QPlainTextEdit(); self._rf_out.setReadOnly(True); self._rf_out.setMinimumHeight(220)
        f = self._rf_out.font(); f.setFamily("Menlo"); f.setPointSize(10); self._rf_out.setFont(f)
        lay.addWidget(self._rf_out)

        # patch log
        g6 = QGroupBox("Patch log"); l6 = QVBoxLayout(g6)
        self._rf_log = QListWidget(); l6.addWidget(self._rf_log)
        row = QHBoxLayout(); b_rv = QPushButton("Revert selected"); b_rv.clicked.connect(self._rf_revert)
        b_rl = QPushButton("Refresh"); b_rl.clicked.connect(self._rf_refresh_log)
        row.addWidget(b_rv); row.addWidget(b_rl); l6.addLayout(row)
        lay.addWidget(g6)
        lay.addStretch(1)
        self._rf_refresh_log()
        return w

    # ── helpers ────────────────────────────────────────────────────────────────────────────────────────────────────
    def _rf_say(self, s, append=True):
        if append:
            self._rf_out.appendPlainText(s)
        else:
            self._rf_out.setPlainText(s)

    def _rf_stage_root(self) -> Path:
        return Path(self.graph_path).parent.parent

    def _rf_mask_path(self) -> Path:
        return self._rf_stage_root() / "mask" / "mask.tif"

    def _rf_patchlog(self):
        from ..edit.patches import PatchLog
        return PatchLog(Path(self.graph_path), self._rf_mask_path())

    def _rf_stage_mask(self):
        import tifffile
        return tifffile.imread(str(self._rf_mask_path())) > 127

    def _rf_ensure_layer(self):
        lyr = getattr(self, "_rf_region_layer", None)
        if lyr is not None and lyr in self.viewer.layers:
            return lyr
        lyr = self.viewer.add_shapes([], name=REFINE_REGION, edge_color="#ffffff", face_color="transparent",
                                     edge_width=2.5, opacity=0.95)
        self._rf_region_layer = lyr
        from qtpy.QtCore import QTimer

        def _on_data(event):
            if str(getattr(event, "action", "")) == "added":
                QTimer.singleShot(0, self._rf_region_added)
        lyr.events.data.connect(_on_data)
        return lyr

    # ── 1 region ───────────────────────────────────────────────────────────────────────────────────────────────────
    def _rf_start_draw(self):
        if not self._require_workbench("refinement"):
            return
        lyr = self._rf_ensure_layer()
        try:
            lyr.data = []
            idx = self.viewer.layers.index(lyr)
            if idx != len(self.viewer.layers) - 1:
                self.viewer.layers.move(idx, len(self.viewer.layers) - 1)
            self.viewer.layers.selection.active = lyr
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()
            lyr.mode = self._rf_shape.currentData()
        except Exception as e:
            self._rf_say(f"could not arm drawing: {e}")
        self._rf_say("Draw the region on the mosaic (lasso: click-drag-click; polygon: clicks, double-click to close).", append=False)

    def _rf_region_added(self):
        lyr = getattr(self, "_rf_region_layer", None)
        if lyr is None or not len(lyr.data):
            return
        rc = np.asarray(lyr.data[-1], float)[:, -2:]
        if len(rc) < 3:
            return
        xy = np.column_stack([rc[:, 1], self.mosaic_height - rc[:, 0]])         # display (row, col) → graph (x, y_up)
        self._rf["poly"] = xy
        try:
            lyr.data = [rc]; lyr.mode = "pan_zoom"
        except Exception:
            pass
        from matplotlib.path import Path as MplPath
        reg = MplPath(xy); n_in = int(reg.contains_points(self._edge_midpoints).sum()) if len(self._edge_midpoints) else 0
        self._rf_region_lbl.setText(f"region: {len(xy)} vertices, ~{n_in} vessels, "
                                    f"x {xy[:, 0].min():.0f}–{xy[:, 0].max():.0f}, y {xy[:, 1].min():.0f}–{xy[:, 1].max():.0f}")
        self._rf_say("Region set.  Optionally edit the mask (step 2), then Re-vectorise (step 3).")

    def _rf_clear(self):
        for name in (REFINE_REGION, REFINE_PREVIEW):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        self._rf_region_layer = None; self._rf["poly"] = None; self._rf["result"] = None
        self._rf_region_lbl.setText("no region")

    # ── 2 mask ─────────────────────────────────────────────────────────────────────────────────────────────────────
    def _rf_edit_mask(self):
        if not self._require_workbench("mask editing"):
            return
        if REFINE_MASK not in self.viewer.layers:
            m = self._rf_stage_mask().astype(np.uint8)
            lyr = self.viewer.add_labels(m, name=REFINE_MASK, opacity=0.45)
            self._rf["mask_layer"] = lyr
        lyr = self.viewer.layers[REFINE_MASK]
        try:
            self.viewer.layers.selection.active = lyr; lyr.selected_label = 1; lyr.brush_size = 6; lyr.mode = "paint"
        except Exception:
            pass
        self._rf_mask_lbl.setText("mask: editing 'Refine mask' layer (label 1 = vessel; use the eraser or label 0 to remove)")

    def _rf_reset_mask(self):
        if REFINE_MASK in self.viewer.layers:
            self.viewer.layers.remove(REFINE_MASK)
        self._rf["mask_layer"] = None; self._rf_mask_lbl.setText("mask: stage mask (unedited)")

    def _rf_candidate_mask(self, base):
        """Stage mask with the 'Refine mask' edits applied INSIDE the region only (None if no edits there)."""
        if REFINE_MASK not in self.viewer.layers or self._rf["poly"] is None:
            return None
        ed = np.asarray(self.viewer.layers[REFINE_MASK].data) > 0
        ch = ed != base
        if not ch.any():
            return None
        from matplotlib.path import Path as MplPath
        rr, cc = np.nonzero(ch); xy = np.column_stack([cc, self.mosaic_height - rr]).astype(float)
        inside = MplPath(self._rf["poly"]).contains_points(xy)
        if not inside.any():
            return None
        out = base.copy(); out[rr[inside], cc[inside]] = ed[rr[inside], cc[inside]]
        if (~inside).any():
            self._rf_say(f"note: {int((~inside).sum())} edited mask px outside the region are ignored")
        return out

    # ── 3 splice ───────────────────────────────────────────────────────────────────────────────────────────────────
    def _rf_splice(self):
        if not self._require_workbench("re-vectorisation"):
            return
        if self._rf["poly"] is None:
            self._rf_say("Draw a region first (step 1)."); return
        if getattr(self, "_radial_tags_dirty", False):
            self._rf_say("Unsaved region-tag edits in the Edit tab: save or discard them first (the refinement works on "
                         "the graph file)."); return
        try:
            from ..edit.splice import splice_region
            base = self._rf_stage_mask(); cand = self._rf_candidate_mask(base)
            self._rf_say("re-vectorising …", append=False)
            from qtpy.QtWidgets import QApplication
            QApplication.processEvents()
            # splice the graph FILE, not self.G: the viewer adds in-memory / cache attributes to every edge at startup,
            # which would otherwise end up in the patch
            import pickle
            G_src = pickle.loads(Path(self.graph_path).read_bytes())
            res = splice_region(G_src, cand if cand is not None else base, self._rf["poly"])
            self._rf["G_src"] = G_src
        except Exception as e:
            self._rf_say(f"re-vectorisation FAILED: {e}"); return
        self._rf["result"] = res; self._rf["mask_new"] = cand; self._rf["pending"] = None
        self._rf_draw_preview(res)
        r = res.report
        lines = [f"old vessels replaced {r['old_removed']} (cut at the border {r['old_cut']}; measured {r['measured_vessels_replaced']}, "
                 f"{r['recordings_dropped']} recordings)",
                 f"new vessels {r['new_edges']}, new nodes {r['new_nodes']}",
                 f"border joins {r['joins']} of {r['ports_old']} old crossings; new-skeleton spurs pruned {r['pruned_tip_nodes']} nodes",
                 f"mask: {'EDITED inside the region' if cand is not None else 'stage mask (unchanged)'}"]
        if r["dead_ends"]:
            lines.append(f"⚠ {len(r['dead_ends'])} DEAD END(S) (old vessel found no new partner) — yellow ×; fix the mask or redraw")
        if r["orphaned_old_nodes"]:
            lines.append(f"⚠ orphaned old nodes {r['orphaned_old_nodes']}")
        lines.append("Preview: red = replaced, green = new.  Next: step 4 (measure + refit) or Discard.")
        self._rf_say("\n".join(lines))

    def _rf_draw_preview(self, res):
        if REFINE_PREVIEW in self.viewer.layers:
            self.viewer.layers.remove(REFINE_PREVIEW)
        H = self.mosaic_height; data, cols, widths = [], [], []
        G_src = self._rf.get("G_src", self.G)
        for e in res.removed_edges:
            p = np.asarray(G_src.edges[e]["path"], float); data.append(np.column_stack([H - p[:, 1], p[:, 0]])); cols.append("#ff3030"); widths.append(5)
        for e in res.new_edges:
            p = np.asarray(res.G.edges[e]["path"], float); data.append(np.column_stack([H - p[:, 1], p[:, 0]])); cols.append("#39ff14"); widths.append(2)
        for d in res.report["dead_ends"]:
            x, y = d["x"], d["y"]; s = 8
            data += [np.array([[H - y - s, x - s], [H - y + s, x + s]]), np.array([[H - y - s, x + s], [H - y + s, x - s]])]
            cols += ["#ffe600", "#ffe600"]; widths += [3, 3]
        if data:
            self.viewer.add_shapes(data, shape_type="path", edge_color=cols, edge_width=widths, name=REFINE_PREVIEW, opacity=0.9)

    # ── 4 evaluate ─────────────────────────────────────────────────────────────────────────────────────────────────
    def _rf_evaluate(self):
        if not self._require_workbench("refinement"):
            return
        res = self._rf.get("result")
        if res is None:
            self._rf_say("Re-vectorise first (step 3)."); return
        if self._rf.get("proc") is not None:
            self._rf_say("a job is already running"); return
        from ..edit.refine import stage_candidate
        if self._rf.get("pending") is None:
            try:
                self._rf["pending"] = stage_candidate(Path(self.graph_path), res.G, polygon=self._rf["poly"], kind="revectorize",
                                                      note=self._rf_note.text(), report=res.report, mask_new=self._rf.get("mask_new"))
            except Exception as e:
                self._rf_say(f"staging FAILED: {e}"); return
        from qtpy.QtCore import QProcess
        p = QProcess()
        args = ["-u", "-m", "plexusflow.cli", "refine-eval", str(self._rf_stage_root()), str(self._rf["pending"])]
        if not self._rf_measure_chk.isChecked():
            args.append("--no-measure")
        env = p.processEnvironment() if hasattr(p, "processEnvironment") else None
        from qtpy.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment(); env.insert("PLEXUSFLOW_STAGE", str(self._rf_stage_root()))
        src = str(Path(__file__).resolve().parents[2]); env.insert("PYTHONPATH", src + os.pathsep + env.value("PYTHONPATH", ""))
        p.setProcessEnvironment(env); p.setProcessChannelMode(QProcess.MergedChannels)
        logf = Path(self._rf["pending"]) / "job.log"

        def _read():
            s = bytes(p.readAllStandardOutput()).decode(errors="replace")
            if s:
                with open(logf, "a") as fh:
                    fh.write(s)
                for line in s.rstrip().splitlines():
                    if line.strip():
                        self._rf_say(line)

        def _done(code, _status=None):
            _read(); self._rf["proc"] = None; self._rf_eval_btn.setEnabled(True); self._rf_stop_btn.setEnabled(False)
            rj = Path(self._rf["pending"]) / "result.json"
            if code == 0 and rj.exists():
                self._rf_show_result(json.loads(rj.read_text()))
            else:
                self._rf_say(f"job ended with code {code} (log: {logf})")
        p.readyReadStandardOutput.connect(_read); p.finished.connect(_done)
        py = _job_python(); self._rf_say(f"starting job ({py}) …  candidate: {self._rf['pending'].name}")
        p.start(py, args); self._rf["proc"] = p; self._rf_eval_btn.setEnabled(False); self._rf_stop_btn.setEnabled(True)

    def _rf_stop(self):
        p = self._rf.get("proc")
        if p is not None:
            p.kill(); self._rf_say("job stopped")

    def _rf_show_result(self, res):
        from ..edit.refine import format_result
        self._rf["eval"] = res
        self._rf_say("\n── result ──\n" + format_result(res) + "\n\nAccept (step 5) writes it to the stage as a patch.")

    # ── 5 accept / discard ─────────────────────────────────────────────────────────────────────────────────────────
    def _rf_accept(self):
        if not self._require_workbench("accepting a patch"):
            return
        from qtpy.QtWidgets import QMessageBox
        res = self._rf.get("result")
        if res is None:
            self._rf_say("nothing to accept"); return
        if self._rf.get("proc") is not None:
            self._rf_say("wait for the job to finish"); return
        note = self._rf_note.text().strip()
        if not note:
            self._rf_say("Write the reason (from the imaging) in the note field first."); return
        if self._rf.get("pending") is None or not (Path(self._rf["pending"]) / "result.json").exists():
            ok = QMessageBox.question(self.viewer.window._qt_window, "Accept without evaluation?",
                                      "This candidate has not been measured / refitted.  Accept anyway?  "
                                      "(its new vessels stay unmeasured until the next analyze run)")
            if ok != QMessageBox.Yes:
                return
            if self._rf.get("pending") is None:
                from ..edit.refine import stage_candidate
                self._rf["pending"] = stage_candidate(Path(self.graph_path), res.G, polygon=self._rf["poly"], kind="revectorize",
                                                      note=note, report=res.report, mask_new=self._rf.get("mask_new"))
        ok = QMessageBox.question(self.viewer.window._qt_window, "Write to the stage?",
                                  f"Write this patch to\n{self.graph_path}\n(graph{' + mask' if self._rf.get('mask_new') is not None else ''}; "
                                  f"reversible from the patch log)?")
        if ok != QMessageBox.Yes:
            return
        from ..edit.refine import accept
        try:
            entry = accept(self._rf["pending"], mask_path=self._rf_mask_path(), note=note)
        except Exception as e:
            self._rf_say(f"accept FAILED: {e}"); return
        self._rf_say(f"accepted as patch {entry['id']} — reloading the graph")
        self._rf_reset_mask(); self._rf_clear(); self._rf["pending"] = None
        self._rf_reload_graph(); self._rf_refresh_log()

    def _rf_discard(self):
        from ..edit.refine import discard
        pend = self._rf.get("pending")
        if pend is not None and Path(pend).exists():
            try:
                discard(pend)
            except Exception as e:
                self._rf_say(f"discard: {e}")
        self._rf["pending"] = None; self._rf["result"] = None
        if REFINE_PREVIEW in self.viewer.layers:
            self.viewer.layers.remove(REFINE_PREVIEW)
        self._rf_say("candidate discarded (region and mask edits kept)", append=False)

    # ── patch log ──────────────────────────────────────────────────────────────────────────────────────────────────
    def _rf_refresh_log(self):
        try:
            es = self._rf_patchlog().entries()
        except Exception:
            es = []
        self._rf_log.clear()
        for e in es:
            c = e.get("counts", {})
            self._rf_log.addItem(f"{e['id']}  {e['status']:8s}  {e['kind']}  −{c.get('removed_edges', 0)}/+{c.get('added_edges', 0)}  "
                                 f"{e['stamp']}  {e.get('note', '')}")

    def _rf_revert(self):
        if not self._require_workbench("reverting a patch"):
            return
        it = self._rf_log.currentItem()
        if it is None:
            self._rf_say("select a patch in the log"); return
        pid = it.text().split()[0]
        from qtpy.QtWidgets import QMessageBox
        ok = QMessageBox.question(self.viewer.window._qt_window, "Revert patch?", f"Revert patch {pid} on the stage graph (and mask)?")
        if ok != QMessageBox.Yes:
            return
        import pickle
        try:
            G = pickle.loads(Path(self.graph_path).read_bytes())
            self._rf_patchlog().revert(G, pid)
        except Exception as e:
            self._rf_say(f"revert FAILED: {e}"); return
        self._rf_say(f"patch {pid} reverted — reloading the graph"); self._rf_reload_graph(); self._rf_refresh_log()

    def _rf_reload_graph(self):
        """Re-read the graph file and rebuild everything the viewer derived from it."""
        self.G = self._load_graph()
        self._rec = self._load_reconciliation()
        self.edge_list = list(self.G.edges())
        self._edge_midpoints = np.array([[0.5 * (self.G.nodes[u]["x"] + self.G.nodes[v]["x"]),
                                          0.5 * (self.G.nodes[u]["y"] + self.G.nodes[v]["y"])] for (u, v) in self.edge_list])
        for f in ("_precompute_harmonic_classes", "_precompute_measured_dissipation", "_refresh_boundary_layers",
                  "_refresh_edges", "_refresh_nodes"):
            try:
                getattr(self, f)()
            except Exception as e:
                print(f"[refine] {f} after reload: {e}")
        if "Vessel mask" in self.viewer.layers:
            try:
                self.viewer.layers["Vessel mask"].data = self._rf_stage_mask().astype(np.uint8)
            except Exception:
                pass
