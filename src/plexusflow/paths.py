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
