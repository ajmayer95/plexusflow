"""Unified Mask + Vectorize + Edit napari session for plexusflow.

One window. Layers (bottom → top):
    probs     — probability map (colormap='viridis', low opacity)
    mask      — editable labels (napari brush/eraser)
    graph     — shapes overlay of the current vectorization
    video     — per-tile video overlay (on demand, via VideoOverlayManager)

Right-side docks:
    Threshold + paint  — high/low sliders, min_size, close_radius, Lock
    Vectorize          — min_feature_size, prune_order, "Vectorize now"
    Tile video overlay — VideoOverlayManager (existing)
    Save               — one button that writes mask.tif + graph.gpickle

Design notes:
- Threshold sliders redraw the mask live UNLESS 'Lock painted mask' is on.
- Vectorize is a button (O(seconds)); running it wipes the graph overlay
  and replaces it with the fresh graph derived from the current mask.
- Graph editing lives in `net3 edit`; this window shows the current graph
  as a read-only overlay for context. (Detailed graph edits still go
  through `plexusflow edit`.)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from ..config import StageConfig


def launch_unified_editor(cfg: StageConfig) -> int:
    """Open the unified mask+vectorize+preview session on `cfg`."""
    cfg.ensure_dirs()
    prob_path = cfg.fuse_probs_logit_path
    if not prob_path.exists():
        raise FileNotFoundError(
            f"no probability map at {prob_path} — run `plexusflow fuse` first")

    # ── Data loaders (mirror plexus_threshold's conventions) ──────────
    def _load(path: Path) -> np.ndarray:
        im = tifffile.imread(str(path))
        if im.ndim == 3 and im.shape[0] == 1:
            im = im[0]
        return np.asarray(im)

    def _to_prob01(im: np.ndarray) -> np.ndarray:
        im = im.astype(np.float32)
        mn, mx = float(im.min()), float(im.max())
        if mx <= 1.0 + 1e-6 and mn >= 0.0 - 1e-6:
            return np.clip(im, 0.0, 1.0)
        # logit-transformed → sigmoid back
        return 1.0 / (1.0 + np.exp(-im))

    prob = _to_prob01(_load(prob_path))
    coverage = (prob > 0.0)
    intensity = _load(cfg.fuse_intensity_path) \
        if cfg.fuse_intensity_path.exists() else None
    if cfg.mask_path.exists():
        mask0 = (_load(cfg.mask_path) > 127).astype(np.uint8)
    else:
        mask0 = (prob > float(cfg.mask_default_threshold)).astype(np.uint8)
    graph0 = None
    if cfg.graph_path.exists():
        import pickle
        try:
            with open(cfg.graph_path, "rb") as f:
                graph0 = pickle.load(f)
        except Exception as e:
            print(f"[mve] failed to load existing graph {cfg.graph_path}: {e}")

    # ── napari ──────────────────────────────────────────────────────────
    import napari
    from qtpy.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QDoubleSpinBox, QSpinBox, QCheckBox, QGroupBox, QComboBox,
        QTabWidget, QSlider)
    from qtpy.QtCore import Qt

    def _make_float_slider(label_text, initial, lo=0.0, hi=1.0, step=0.001):
        """Draggable horizontal slider + a numeric readout, both wired
        together. Returns (container_widget, get_value, set_value,
        connect_changed)."""
        w = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel(label_text))
        s = QSlider(Qt.Horizontal)
        n_steps = max(1, int(round((hi - lo) / step)))
        s.setRange(0, n_steps); s.setSingleStep(1); s.setPageStep(max(1, n_steps // 20))
        s.setValue(int(round((initial - lo) / step)))
        s.setMinimumWidth(140)
        lay.addWidget(s, stretch=1)
        readout = QLabel(f"{initial:.3f}")
        readout.setMinimumWidth(46)
        lay.addWidget(readout)
        def _get(): return lo + s.value() * step
        def _set(v):
            s.setValue(int(round((float(v) - lo) / step)))
        def _on_change(fn):
            s.valueChanged.connect(lambda _: (readout.setText(f"{_get():.3f}"),
                                                fn(_get())))
        s.valueChanged.connect(lambda _: readout.setText(f"{_get():.3f}"))
        return w, _get, _set, _on_change

    viewer = napari.Viewer(title=f"plexusflow mve · {cfg.root.name}")

    if intensity is not None:
        viewer.add_image(
            intensity, name="intensity", colormap="gray",
            contrast_limits=[float(np.percentile(intensity, 1)),
                             float(np.percentile(intensity, 99.5))],
            blending="translucent", opacity=0.8)
    prob_layer = viewer.add_image(
        prob, name="probability", colormap="viridis",
        contrast_limits=(0.0, 1.0), blending="translucent", opacity=0.4,
        visible=False)
    mask_layer = viewer.add_labels(mask0.astype(np.uint8), name="mask")
    graph_shapes_layer = None
    graph_nodes_layer = None

    # Helpers ──────────────────────────────────────────────────────────
    def _make_mask(hi: float, lo: float, min_size: int, close_radius: int) -> np.ndarray:
        """Threshold + cleanup, matching plexus_threshold's algorithm."""
        from skimage import filters, morphology
        if lo > 0.0 and lo < hi:
            m = filters.apply_hysteresis_threshold(prob, lo, hi)
        else:
            m = prob >= hi
        m = m & coverage
        if close_radius > 0:
            m = morphology.binary_closing(m, morphology.disk(close_radius))
        if min_size > 0:
            m = morphology.remove_small_objects(m, min_size=min_size)
            m = morphology.remove_small_holes(m, area_threshold=min_size)
        return m.astype(np.uint8)

    def _draw_graph(G):
        nonlocal graph_shapes_layer, graph_nodes_layer
        for lyr in (graph_shapes_layer, graph_nodes_layer):
            if lyr is not None:
                try: viewer.layers.remove(lyr)
                except Exception: pass
        graph_shapes_layer = None; graph_nodes_layer = None
        if G is None or G.number_of_edges() == 0:
            return
        # net3 / plexusflow convention: graph node y is measured from the
        # BOTTOM of the mask image, so drawing on top of the mask needs
        # row = image_h - y to appear right-side up.
        H_img = int(mask_layer.data.shape[0])
        segs = []
        for u, v in G.edges():
            pu, pv = G.nodes[u], G.nodes[v]
            xu, yu = pu.get("x"), pu.get("y")
            xv, yv = pv.get("x"), pv.get("y")
            if None in (xu, yu, xv, yv): continue
            segs.append([[H_img - float(yu), float(xu)],
                          [H_img - float(yv), float(xv)]])
        if segs:
            graph_shapes_layer = viewer.add_shapes(
                segs, shape_type="line", edge_color="#ffb700",
                edge_width=1.2, name="graph edges", opacity=0.9)
        pts = []; colors = []; sizes = []
        n_tips = 0
        for n, d in G.nodes(data=True):
            x, y = d.get("x"), d.get("y")
            if x is None or y is None: continue
            pts.append([H_img - float(y), float(x)])
            if G.degree(n) == 1:
                colors.append("#00ffe1")   # cyan for tips (endpoints)
                sizes.append(9)
                n_tips += 1
            else:
                colors.append("#ff5555")   # red for junctions/interior
                sizes.append(3)
        if pts:
            graph_nodes_layer = viewer.add_points(
                np.asarray(pts), name="graph nodes",
                size=np.asarray(sizes), face_color=colors, opacity=0.9,
                border_color="black", border_width=0.3)
        print(f"[mve] graph overlay: {G.number_of_nodes()} nodes "
              f"({n_tips} tips cyan), {G.number_of_edges()} edges  "
              f"(y-flipped by H={H_img})")

    if graph0 is not None:
        _draw_graph(graph0)

    # ── Tabbed dock: Mask / Vectorize ─────────────────────────────────
    tabs = QTabWidget()

    # -- Mask tab --
    thr_box = QWidget()
    tlay = QVBoxLayout(thr_box)
    hi_w, hi_get, hi_set, hi_conn = _make_float_slider(
        "High:", float(cfg.mask_default_threshold), 0.0, 1.0, 0.001)
    tlay.addWidget(hi_w)
    lo_w, lo_get, lo_set, lo_conn = _make_float_slider(
        "Low  :", float(cfg.mask_default_low), 0.0, 1.0, 0.001)
    tlay.addWidget(lo_w)
    class _SliderAdapter:
        """Give the sliders a QDoubleSpinBox-like API so the rest of the
        code (valueChanged.connect, .value(), .setEnabled) keeps working."""
        def __init__(self, w, get_fn, set_fn, conn_fn):
            self._w = w; self._get = get_fn; self._set = set_fn
            self._conn = conn_fn
        def value(self): return self._get()
        def setValue(self, v): self._set(v)
        def setEnabled(self, b): self._w.setEnabled(bool(b))
        class _ValueChanged:
            def __init__(self, adapter): self._a = adapter
            def connect(self, fn): self._a._conn(lambda v: fn(v))
        @property
        def valueChanged(self): return self._ValueChanged(self)
    hi_spin = _SliderAdapter(hi_w, hi_get, hi_set, hi_conn)
    lo_spin = _SliderAdapter(lo_w, lo_get, lo_set, lo_conn)
    tlay.addWidget(QLabel("Min feature / hole size (px):"))
    ms_spin = QSpinBox(); ms_spin.setRange(0, 100_000)
    ms_spin.setValue(int(cfg.mask_default_min_size)); tlay.addWidget(ms_spin)
    tlay.addWidget(QLabel("Closing radius (px):"))
    cr_spin = QSpinBox(); cr_spin.setRange(0, 20)
    cr_spin.setValue(int(cfg.mask_default_close_radius)); tlay.addWidget(cr_spin)
    lock_chk = QCheckBox(
        "Freeze mask entirely (Vectorize mode auto-locks)")
    lock_chk.setChecked(cfg.mask_path.exists())
    tlay.addWidget(lock_chk)
    keep_paint_chk = QCheckBox("Keep painted edits when re-thresholding")
    keep_paint_chk.setChecked(True)
    tlay.addWidget(keep_paint_chk)
    clear_paint_btn = QPushButton("Clear painted edits")
    clear_paint_btn.setToolTip(
        "Discard all manual brush/eraser edits. Mask reverts to the pure "
        "threshold output with the current slider values.")
    tlay.addWidget(clear_paint_btn)
    thr_status = QLabel("")
    tlay.addWidget(thr_status)
    tlay.addStretch(1)
    tabs.addTab(thr_box, "Mask")

    # Auto (threshold) mask kept separately from the composite so we can
    # apply a manual paint diff on top of a freshly re-thresholded mask.
    _auto = {"mask": None}
    def _recompute_auto():
        _auto["mask"] = _make_mask(hi_spin.value(), lo_spin.value(),
                                    ms_spin.value(), cr_spin.value())

    def _paint_diff():
        """Return the manual delta: current mask - auto mask, as int8."""
        cur = np.asarray(mask_layer.data, dtype=np.int8)
        if _auto["mask"] is None: _recompute_auto()
        return cur - _auto["mask"].astype(np.int8)

    def _composite_current():
        """Auto ∪ painted-on, then remove painted-off, then clip to coverage."""
        auto = _auto["mask"]
        diff = _paint_diff_saved["arr"]
        if diff is None:
            m = auto.copy()
        else:
            m = np.clip(auto.astype(np.int16) + diff.astype(np.int16), 0, 1)
        m = (m > 0) & coverage
        return m.astype(np.uint8)

    _paint_diff_saved = {"arr": None}   # int8 (H, W) or None

    def _re_threshold(*_):
        if lock_chk.isChecked():
            thr_status.setText("(mask frozen)")
            return
        # 1) Snapshot the current manual diff BEFORE we swap the auto mask.
        if keep_paint_chk.isChecked() and _auto["mask"] is not None:
            _paint_diff_saved["arr"] = _paint_diff()
        else:
            _paint_diff_saved["arr"] = None
        # 2) Recompute auto with the new slider values.
        _recompute_auto()
        # 3) Overlay the paint diff and push to the layer.
        m = _composite_current()
        mask_layer.data = m
        n_edit = int(np.abs(_paint_diff_saved["arr"]).sum()) \
            if _paint_diff_saved["arr"] is not None else 0
        thr_status.setText(
            f"mask: {int(m.sum()):,} px  ({n_edit:,} px manual)")

    for w in (hi_spin, lo_spin, ms_spin, cr_spin):
        w.valueChanged.connect(_re_threshold)

    def _clear_painted():
        _paint_diff_saved["arr"] = None
        _recompute_auto()
        mask_layer.data = _composite_current()
        thr_status.setText(
            f"cleared manual edits — mask: {int(mask_layer.data.sum()):,} px")
    clear_paint_btn.clicked.connect(_clear_painted)

    if not cfg.mask_path.exists():
        _re_threshold()
    else:
        _recompute_auto()
        # Treat the loaded mask as an all-manual-diff over the current
        # auto so re-thresholding won't destroy what the user already
        # has on disk (until they choose to Clear).
        _paint_diff_saved["arr"] = (
            mask0.astype(np.int8) - _auto["mask"].astype(np.int8))
        thr_status.setText(
            f"loaded existing mask: {int((mask0>0).sum()):,} px "
            f"({int(np.abs(_paint_diff_saved['arr']).sum()):,} px vs auto)")

    # -- Vectorize tab --
    vec_box = QWidget()
    vlay = QVBoxLayout(vec_box)
    vlay.addWidget(QLabel("Min feature size (px):"))
    mfs_spin = QSpinBox(); mfs_spin.setRange(0, 1_000_000)
    mfs_spin.setValue(int(cfg.vectorize_min_feature_size)); vlay.addWidget(mfs_spin)
    vlay.addWidget(QLabel("Prune order (triangles from tip):"))
    po_spin = QSpinBox(); po_spin.setRange(0, 20)
    po_spin.setValue(int(cfg.vectorize_prune_order)); vlay.addWidget(po_spin)
    vlay.addWidget(QLabel("Fill holes ≤ (px area):"))
    fh_spin = QSpinBox(); fh_spin.setRange(0, 100_000)
    fh_spin.setValue(int(cfg.vectorize_fill_holes)); vlay.addWidget(fh_spin)
    vlay.addWidget(QLabel("Merge junctions within (px, 0 = off):"))
    mj_spin = QDoubleSpinBox(); mj_spin.setRange(0.0, 100.0)
    mj_spin.setSingleStep(0.5); mj_spin.setDecimals(1)
    mj_spin.setValue(3.0)
    mj_spin.setToolTip(
        "Union-find over junctions (deg≥3) within this pixel distance. "
        "Each cluster becomes a single node at the centroid; incident "
        "edges are re-attached, paths' endpoints are snapped to the new "
        "centroid so connectivity + polyline shape are preserved.")
    vlay.addWidget(mj_spin)
    vec_btn = QPushButton("Vectorize now")
    vlay.addWidget(vec_btn)
    vec_status = QLabel("(graph shown only for reference; edit via `plexusflow edit`)")
    vec_status.setWordWrap(True)
    vlay.addWidget(vec_status)
    vlay.addStretch(1)
    tabs.addTab(vec_box, "Vectorize")

    viewer.window.add_dock_widget(tabs, area="right", name="Mask / Vectorize")

    # ── Tab-as-mode interlocks ────────────────────────────────────────
    # Track whether the mask has been edited since the last vectorize —
    # drives the "Vectorize ⚠ stale" tab title and dims the graph.
    _state = {"mask_dirty": False, "graph_dirty": False}
    _mask_widgets = (hi_spin, lo_spin, ms_spin, cr_spin, lock_chk,
                     keep_paint_chk, clear_paint_btn)
    _vec_widgets  = (mfs_spin, po_spin, fh_spin, mj_spin, vec_btn)

    def _refresh_tab_titles():
        tabs.setTabText(0, "Mask" + (" ⚠" if _state["graph_dirty"] else ""))
        tabs.setTabText(1, "Vectorize" + (" ⚠" if _state["mask_dirty"] else ""))

    def _refresh_save_labels():
        stale_m = " ⚠" if _state["mask_dirty"] else ""
        stale_g = " ⚠" if _state["graph_dirty"] else ""
        save_mask_btn.setText(f"Save mask{stale_m} → {cfg.mask_path.name}")
        save_graph_btn.setText(f"Save graph{stale_g} → {cfg.graph_path.name}")

    def _apply_mode(idx: int):
        """0 = Mask edit, 1 = Vectorize (mask frozen)."""
        mask_mode = (idx == 0)
        # Threshold widgets enabled only in Mask mode
        for w in _mask_widgets: w.setEnabled(mask_mode)
        # Vectorize widgets enabled only in Vectorize mode
        for w in _vec_widgets: w.setEnabled(not mask_mode)
        # Freeze the paint brush by switching into pan/zoom in Vectorize
        try:
            mask_layer.mode = "paint" if mask_mode else "pan_zoom"
        except Exception:
            pass
        # Fade the graph in Mask mode (context only), full in Vectorize
        for gl in (graph_shapes_layer, graph_nodes_layer):
            if gl is not None:
                try: gl.opacity = 0.35 if mask_mode else 0.9
                except Exception: pass
    tabs.currentChanged.connect(_apply_mode)

    def _on_mask_paint(_event=None):
        _state["mask_dirty"] = True
        # Capture the new paint diff so a later re-threshold preserves it.
        try:
            if _auto["mask"] is not None and keep_paint_chk.isChecked():
                _paint_diff_saved["arr"] = _paint_diff()
        except Exception:
            pass
        # If a graph exists, mark it stale — it no longer reflects the mask.
        if _current[0] is not None:
            _state["graph_dirty"] = True
        _refresh_tab_titles(); _refresh_save_labels()
    try:
        mask_layer.events.paint.connect(_on_mask_paint)
    except Exception:
        try: mask_layer.events.data.connect(_on_mask_paint)
        except Exception: pass

    def _run_vectorize():
        from net3.pipeline import vectorize as _vec
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "mask_current.tif"
            arr = np.asarray(mask_layer.data)
            tifffile.imwrite(str(tmp), (arr > 0).astype(np.uint8) * 255)
            try:
                G = _vec(str(tmp),
                        min_feature_size=int(mfs_spin.value()),
                        prune_order=int(po_spin.value()),
                        merge_junctions_d_max=float(mj_spin.value()),
                        # Keep degree-2 skeleton nodes so downstream
                        # `condense` can walk them into per-edge
                        # centerline paths. Without this, paths degrade
                        # to just [start, end].
                        remove_redundant="none",
                        verbose=False)
            except Exception as e:
                vec_status.setText(f"vectorize FAILED: {e}")
                print(f"[mve] vectorize failed: {e!r}")
                return
        _draw_graph(G)
        vec_status.setText(f"vectorized: {G.number_of_nodes()} nodes, "
                           f"{G.number_of_edges()} edges")
        _current[0] = G   # remember for save
        _state["mask_dirty"] = False
        _state["graph_dirty"] = True   # newly created; not yet saved
        _refresh_tab_titles(); _refresh_save_labels()
        # Bring the fresh graph to full opacity (we're in Vectorize mode).
        for gl in (graph_shapes_layer, graph_nodes_layer):
            if gl is not None:
                try: gl.opacity = 0.9
                except Exception: pass
    vec_btn.clicked.connect(_run_vectorize)
    _current = [graph0]

    # ── Video overlay dock (reuse the shared manager) ─────────────────
    if cfg.positions_path.exists():
        video_dir = None
        if cfg.videos_realtime_dir.exists() and any(cfg.videos_realtime_dir.iterdir()):
            video_dir = cfg.videos_realtime_dir
        elif cfg.raw_video_dir.exists():
            video_dir = cfg.raw_video_dir
        if video_dir is not None:
            try:
                from net3.edit.video_overlay import (
                    VideoOverlayManager, load_tile_positions)
                positions = load_tile_positions(cfg.positions_path)
                mgr = VideoOverlayManager(
                    viewer, positions, [video_dir], cfg.raw_video_dir_pattern,
                    mask_shape=tuple(int(x) for x in mask_layer.data.shape[:2]))
                vdock = QWidget(); vl = QVBoxLayout(vdock)
                vl.addWidget(QLabel("<b>Tile video overlay</b>"))
                row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0,0,0,0)
                rl.addWidget(QLabel("Tile:"))
                combo = QComboBox()
                for tid in mgr.available_tile_ids():
                    combo.addItem(str(tid), userData=tid)
                rl.addWidget(combo, stretch=1); vl.addWidget(row)
                load_btn = QPushButton("Load tile video")
                load_btn.clicked.connect(
                    lambda: mgr.load_tile(int(combo.currentData()), opacity=0.75))
                vl.addWidget(load_btn)
                unload_btn = QPushButton("Unload video"); vl.addWidget(unload_btn)
                unload_btn.clicked.connect(mgr.unload)
                b_chk = QCheckBox("Show tile boundaries")
                b_chk.toggled.connect(lambda c: mgr.set_show_tile_boundaries(bool(c)))
                vl.addWidget(b_chk)
                l_chk = QCheckBox("Show tile numbers")
                l_chk.toggled.connect(lambda c: mgr.set_show_tile_numbers(bool(c)))
                vl.addWidget(l_chk)
                vl.addStretch(1)
                viewer.window.add_dock_widget(vdock, area="right",
                                               name="Tile video overlay")
                print(f"[mve] video overlay ready: {len(positions)} tiles")
            except Exception as e:
                print(f"[mve] video overlay skipped: {e}")

    # ── Save dock ─────────────────────────────────────────────────────
    save_box = QGroupBox("Save")
    slay = QVBoxLayout(save_box)
    save_mask_btn = QPushButton(f"Save mask → {cfg.mask_path.name}")
    slay.addWidget(save_mask_btn)
    save_graph_btn = QPushButton(f"Save graph → {cfg.graph_path.name}")
    slay.addWidget(save_graph_btn)
    save_status = QLabel(""); slay.addWidget(save_status)
    viewer.window.add_dock_widget(save_box, area="right", name="Save")

    def _save_mask():
        arr = np.asarray(mask_layer.data)
        tifffile.imwrite(str(cfg.mask_path),
                         (arr > 0).astype(np.uint8) * 255)
        save_status.setText(f"saved mask ({int(arr.sum()):,} px) → {cfg.mask_path.name}")
        print(f"[mve] wrote {cfg.mask_path}")
        _state["mask_dirty"] = False
        _refresh_tab_titles(); _refresh_save_labels()
    def _save_graph():
        if _current[0] is None:
            save_status.setText("no graph to save — click Vectorize first")
            return
        import pickle
        cfg.graph_path.parent.mkdir(exist_ok=True, parents=True)
        with open(cfg.graph_path, "wb") as f:
            pickle.dump(_current[0], f)
        save_status.setText(f"saved graph ({_current[0].number_of_nodes()} nodes) "
                            f"→ {cfg.graph_path.name}")
        print(f"[mve] wrote {cfg.graph_path}")
        _state["graph_dirty"] = False
        _refresh_tab_titles(); _refresh_save_labels()
    save_mask_btn.clicked.connect(_save_mask)
    save_graph_btn.clicked.connect(_save_graph)

    # Apply initial mode after everything (incl. save buttons) exists
    _apply_mode(0); _refresh_tab_titles(); _refresh_save_labels()

    print("\n=== plexusflow mve ===")
    print("  threshold sliders redraw mask live (unless Lock is on)")
    print("  paint on the mask layer with napari's brush/eraser")
    print("  Vectorize button rebuilds graph from current mask")
    print("  Save writes mask.tif and graph.gpickle to their stage paths")
    napari.run()
    return 0
