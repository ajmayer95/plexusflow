"""Reading data written under earlier names.  The ONLY module that knows the legacy names.

Stage data written before 2026-11-24 names the per-vessel flux estimator (parabolic profile fits per cross-section)
after an author; current code writes and reads only the `profile` names.  Everything that loads stored data passes it
through here, so old graphs, sidecars and releases keep working and a new batch never needs a migration:

  graph     `upgrade_graph(G)` renames per-recording keys in place (idempotent; a graph already in the new names is
            unchanged).  Called by `plexusflow.paths.read_graph` / `load_graph`, the viewer and the batch.
  sidecars  `sidecar_dir(stage)` → `<stage>/batch_profile_inputs`, or the legacy folder when only that exists;
            `sidecar_inputs(d)` → the per-edge inputs of one sidecar file under either key.
  migrate   `python -m plexusflow.compat <stage>` rewrites the stage graph in the new names (backup first) and renames
            the sidecar folder; optional — reading works without it.
"""
from __future__ import annotations

from pathlib import Path

# per-recording keys (measurements_piv entries)
MEASUREMENT_KEYS = {
    "Q_kl": "Q_profile",
    "R_kl_um": "R_profile_um",
    "V_CL_kl_ums": "V_CL_profile_ums",
    "Q_kl_valid_frac": "Q_profile_valid_frac",
    "Q_kl_n_valid": "Q_profile_n_valid",
}
QUALITY_KEYS = {"kl_ok": "profile_ok"}            # measurements_piv[*]['quality']
PROV_KEYS = {"kl_mode": "profile_mode"}           # measurements_piv[*]['prov']
BEST_KEY = {"kl": "profile"}                           # measurements_piv[*]['Q_nls']['best_key']
BATCH_KEYS = {"kl_sidecar": "profile_sidecar"}    # G.graph['measurement_batch']
SIDECAR_DIR, LEGACY_SIDECAR_DIR = "batch_profile_inputs", "batch_kl_inputs"
SIDECAR_KEY, LEGACY_SIDECAR_KEY = "profile_inputs", "kl_inputs"


def _rename(d: dict, mapping: dict) -> int:
    n = 0
    for old, new in mapping.items():
        if old in d:
            v = d.pop(old)
            d.setdefault(new, v)
            n += 1
    return n


def upgrade_measurement(m: dict) -> int:
    """Rename legacy keys of one measurement dict in place; returns the number of keys renamed."""
    if not isinstance(m, dict):
        return 0
    n = _rename(m, MEASUREMENT_KEYS)
    for sub, mapping in (("quality", QUALITY_KEYS), ("prov", PROV_KEYS)):
        if isinstance(m.get(sub), dict):
            n += _rename(m[sub], mapping)
    nls = m.get("Q_nls")
    if isinstance(nls, dict) and nls.get("best_key") in BEST_KEY:
        nls["best_key"] = BEST_KEY[nls["best_key"]]
        n += 1
    return n


def upgrade_graph(G) -> int:
    """Rename legacy keys throughout a measured graph, in place.  Returns the number of keys renamed (0 = already current)."""
    n = 0
    for *_, d in G.edges(data=True):
        for m in d.get("measurements_piv") or []:
            n += upgrade_measurement(m)
    mb = G.graph.get("measurement_batch")
    if isinstance(mb, dict):
        n += _rename(mb, BATCH_KEYS)
    return n


def loads_graph(data: bytes):
    """Unpickle a measured graph and bring it to the current names (`upgrade_graph`)."""
    import pickle
    G = pickle.loads(data)
    upgrade_graph(G)
    return G


def sidecar_dir(stage) -> Path:
    """Per-tile sidecar folder of a stage: the current name, or the legacy one when only that exists."""
    stage = Path(stage)
    new, old = stage / SIDECAR_DIR, stage / LEGACY_SIDECAR_DIR
    return old if (not new.exists() and old.exists()) else new


def sidecar_inputs(d: dict) -> dict:
    """Per-edge inputs stored in one sidecar file (current or legacy key)."""
    return d.get(SIDECAR_KEY, d.get(LEGACY_SIDECAR_KEY, {}))


def migrate_stage(stage) -> None:
    """Rewrite <stage>/graph/graph_vessels_piv.gpickle in the current names (backup next to it) and rename the
    sidecar folder.  Run with numpy < 2 (system python3) so every environment can still read the graph."""
    import pickle, datetime
    import numpy as np
    if int(np.__version__.split(".")[0]) >= 2:
        raise SystemExit("run with numpy < 2 (system python3): graphs must stay numpy-1 pickles")
    stage = Path(stage); gp = stage / "graph" / "graph_vessels_piv.gpickle"
    G = pickle.loads(gp.read_bytes()); n = upgrade_graph(G)
    if n:
        bak = gp.with_name(gp.name + f".pre_profile_names_{datetime.date.today():%Y%m%d}")
        bak.write_bytes(gp.read_bytes())
        tmp = gp.with_suffix(".tmp"); tmp.write_bytes(pickle.dumps(G, protocol=4)); tmp.replace(gp)
        print(f"graph: {n} keys renamed (backup {bak.name})")
    else:
        print("graph: already in the current names")
    old, new = stage / LEGACY_SIDECAR_DIR, stage / SIDECAR_DIR
    if old.exists() and not new.exists():
        old.rename(new); print(f"sidecars: {old.name}/ → {new.name}/ (files keep their contents; read under either key)")


if __name__ == "__main__":
    import sys
    migrate_stage(sys.argv[1])
