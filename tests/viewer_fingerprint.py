"""Headless fingerprint of the napari viewer on a data release (used by tests/test_viewer_smoke.py and to check that
refactors of plexusflow.viewer do not change behaviour).

    python tests/viewer_fingerprint.py /path/to/release [out.json]

Builds the viewer (view mode), its panel and every tab, then colours the network by each discoverable field and
records layer names, panel tab titles and a hash of the edge colours per field.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def fingerprint(release, max_fields=200, mode="view"):
    import numpy as np
    import napari
    _V = napari.Viewer
    napari.Viewer = lambda *a, **k: _V(*a, **{**k, "show": False})     # build every widget, never show a window
    from plexusflow.viewer.app import ReadOnlyMosaicViewer
    rel = Path(release)
    mosaic = next((p for p in (rel / "fuse" / "organ_fusion_close.tif", rel / "fuse" / "organ_fusion.tif") if p.exists()), None)
    v = ReadOnlyMosaicViewer(rel / "graph" / "graph_vessels_piv.gpickle", tiff_path=mosaic,
                             tile_positions=rel / "stitch" / "positions.json", mode=mode)
    v._setup_viewer(); v._setup_panel(); v._refresh_edges(); v._refresh_nodes(); v._refresh_cbar()
    out = dict(layers=[ly.name for ly in v.viewer.layers])
    try:
        from qtpy.QtWidgets import QTabWidget
        tabs = [w for w in v.viewer.window._qt_window.findChildren(QTabWidget)]
        out["tabs"] = sorted({w.tabText(i) for w in tabs for i in range(w.count())})
    except Exception as e:                                       # pragma: no cover - diagnostic only
        out["tabs"] = f"unavailable: {e}"
    from plexusflow.viewer.app import PROPERTY_DEFS, HARMONIC_KEYS
    combos = [(src, q, key, h) for key, _, harm, valid in PROPERTY_DEFS for (src, q) in sorted(valid)
              if src in ("measured", "reconciled") for h in (HARMONIC_KEYS if harm else [None])]
    out["n_fields"] = len(combos)
    colours = {}
    for src, q, key, h in combos[:max_fields]:
        name = f"{src}:{q}.{key}" + (f"@{h}" if h else "")
        try:
            v.current_source, v.current_quantity, v.current_property = src, q, key
            if h:
                v.current_harmonic = h
            v._refresh_edges()
            ly = next(ly for ly in v.viewer.layers if ly.name == "Edges")
            arr = np.asarray(ly.edge_color, float)
            colours[name] = hashlib.sha1(np.round(arr, 4).tobytes()).hexdigest()[:12]
        except Exception as e:
            colours[name] = f"error: {type(e).__name__}: {e}"
    out["edge_colours"] = colours
    v.viewer.close()
    return out


if __name__ == "__main__":
    fp = fingerprint(sys.argv[1], mode=os.environ.get("VIEWER_MODE", "view"))
    txt = json.dumps(fp, indent=1, sort_keys=True)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(txt)
    print(txt[:3000])
