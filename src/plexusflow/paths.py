"""Machine-independent locations of the stage data and derived results.

The stage folder (graph, Kloosterman sidecars, figures, results) lives outside the repo — typically in a
synced drive — and its path differs between machines.  Set it once per machine:

    export PLEXUSFLOW_STAGE="/path/to/PlexusFlow21S"        # e.g. in ~/.zshrc

Derived results (reconciliation, AC fields, fits) go to ``RESULTS`` = ``$PLEXUSFLOW_RESULTS`` or
``<stage>/results`` so that every machine sees the same outputs; figures to ``FIGURES`` = ``$PLEXUSFLOW_FIGURES`` or
``<stage>/figures``.  When working from a frozen release, point RESULTS and FIGURES at your own folders so the release
stays untouched.  ``MOSAIC`` is the organ mosaic used as figure background.  Nothing here touches data in git.

There is no default stage: using any of these paths without ``PLEXUSFLOW_STAGE`` raises ``StageNotSetError`` (so a
script can never silently read or overwrite another dataset).  Paths are resolved when first accessed.
"""
from __future__ import annotations
import os
from pathlib import Path

__all__ = ["STAGE", "RESULTS", "GRAPH", "FIGURES", "MOSAIC", "load_graph", "attach_external_traces", "StageNotSetError"]


class StageNotSetError(RuntimeError):
    """Raised when a stage path is needed but ``PLEXUSFLOW_STAGE`` is not set (there is deliberately no default)."""


def _stage() -> Path:
    v = os.environ.get("PLEXUSFLOW_STAGE")
    if not v:
        raise StageNotSetError(
            "PLEXUSFLOW_STAGE is not set. Point it at a stage or release folder, e.g.\n"
            '    export PLEXUSFLOW_STAGE="$HOME/data/PlexusFlow21S_release_v2"\n'
            "(see README, 'Reproducing the paper').")
    p = Path(v).expanduser()
    if not p.is_dir():
        raise StageNotSetError(f"PLEXUSFLOW_STAGE={v!r} is not a directory.")
    return p


def _resolve(name: str) -> Path:
    if name == "STAGE":
        return _stage()
    if name == "RESULTS":
        return Path(os.environ.get("PLEXUSFLOW_RESULTS", _stage() / "results")).expanduser()
    if name == "FIGURES":
        return Path(os.environ.get("PLEXUSFLOW_FIGURES", _stage() / "figures")).expanduser()
    if name == "GRAPH":
        return _stage() / "graph" / "graph_vessels_piv.gpickle"
    if name == "MOSAIC":
        st = _stage()
        return next((p for p in (st / "fuse" / "organ_fusion_close.tif", st / "fuse" / "organ_fusion.tif") if p.exists()),
                    st / "fuse" / "organ_fusion_close.tif")
    raise AttributeError(f"module 'plexusflow.paths' has no attribute {name!r}")


def __getattr__(name: str) -> Path:        # PEP 562: paths are resolved on first use, so importing never needs data
    return _resolve(name)

def load_graph(path=None):
    """Load the measured graph (networkx gpickle).  Graphs are pickled with numpy 1.x so that every environment can read
    them; numpy 2 reads them fine but warns about the old ``numpy.core`` module path — silenced here.  Release graphs
    keep their Q(t) traces in export/traces.npz; they are re-attached here (`attach_external_traces`)."""
    import pickle, warnings
    p = Path(path) if path else _resolve("GRAPH")
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
