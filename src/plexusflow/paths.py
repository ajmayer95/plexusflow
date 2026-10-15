"""Machine-independent locations of the stage data and derived results.

The stage folder (graph, Kloosterman sidecars, figures, results) lives outside the repo — typically in a
synced drive — and its path differs between machines.  Set it once per machine:

    export PLEXUSFLOW_STAGE="/path/to/PlexusFlow21S"        # e.g. in ~/.zshrc

Derived results (reconciliation, AC fields, fits) go to ``RESULTS`` = ``$PLEXUSFLOW_RESULTS`` or
``<stage>/results`` so that every machine sees the same outputs; figures to ``FIGURES`` = ``$PLEXUSFLOW_FIGURES`` or
``<stage>/figures``.  When working from a frozen release, point RESULTS and FIGURES at your own folders so the release
stays untouched.  ``MOSAIC`` is the organ mosaic used as figure background.  Nothing here touches data in git.
"""
from __future__ import annotations
import os
from pathlib import Path

_DEFAULT_STAGE = ("/Users/amayer/Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/"
                  "My Drive/PlexusFlow21S")

STAGE = Path(os.environ.get("PLEXUSFLOW_STAGE", _DEFAULT_STAGE)).expanduser()
RESULTS = Path(os.environ.get("PLEXUSFLOW_RESULTS", STAGE / "results")).expanduser()
GRAPH = STAGE / "graph" / "graph_vessels_piv.gpickle"
FIGURES = Path(os.environ.get("PLEXUSFLOW_FIGURES", STAGE / "figures")).expanduser()
MOSAIC = next((p for p in (STAGE / "fuse" / "organ_fusion_close.tif", STAGE / "fuse" / "organ_fusion.tif") if p.exists()),
              STAGE / "fuse" / "organ_fusion_close.tif")


def load_graph(path=None):
    """Load the measured graph (networkx gpickle).  Graphs are pickled with numpy 1.x so that every environment can read
    them; numpy 2 reads them fine but warns about the old ``numpy.core`` module path — silenced here.  Release graphs
    keep their Q(t) traces in export/traces.npz; they are re-attached here (`attach_external_traces`)."""
    import pickle, warnings
    p = Path(path or GRAPH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        G = pickle.loads(p.read_bytes())
    return attach_external_traces(G, p)


def attach_external_traces(G, graph_path):
    """Release graphs store each measurement's Q(t) once, in ``G.graph['external_traces']`` (an npz next to the graph,
    key ``m<meas_id>``), not inside the pickle.  Put them back on the measurements so all code sees ``m['Q_t']``."""
    rel = G.graph.get("external_traces")
    if not rel:
        return G
    import numpy as np
    Z = np.load(Path(graph_path).parent / rel)
    for *_, d in G.edges(data=True):
        for m in d.get("measurements_piv") or []:
            if m.get("Q_t") is None and m.get("meas_id") is not None:
                k = f"m{m['meas_id']}"
                if k in Z.files:
                    m["Q_t"] = Z[k]
    return G
