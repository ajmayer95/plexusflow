"""Viewer structure and (when napari and a data release are available) a headless build of every tab."""
import importlib.util
import os
from pathlib import Path

import pytest


def test_viewer_imports_without_napari_and_keeps_its_methods():
    from plexusflow.viewer.app import ReadOnlyMosaicViewer
    from plexusflow.viewer.simulate import SimulateMixin
    from plexusflow.viewer.inference import InferenceMixin
    assert issubclass(ReadOnlyMosaicViewer, SimulateMixin) and issubclass(ReadOnlyMosaicViewer, InferenceMixin)
    for m in ("_setup_panel", "_refresh_edges", "_run_simulation", "_inference_run", "_on_run_optical_flow_clicked",
              "_plot_selected_qt", "_save_graph_edits", "run"):
        assert callable(getattr(ReadOnlyMosaicViewer, m)), m


@pytest.mark.skipif(importlib.util.find_spec("napari") is None or not os.environ.get("PLEXUSFLOW_TEST_RELEASE"),
                    reason="needs napari and PLEXUSFLOW_TEST_RELEASE (an extracted data release)")
def test_viewer_builds_every_field_headless():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from viewer_fingerprint import fingerprint
    fp = fingerprint(os.environ["PLEXUSFLOW_TEST_RELEASE"])
    assert {"Edges", "Nodes"} <= set(fp["layers"])
    assert {"View", "Time series"} <= set(fp["tabs"])
    errors = {k: v for k, v in fp["edge_colours"].items() if v.startswith("error")}
    assert not errors, errors
