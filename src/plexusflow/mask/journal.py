"""Mask edit journal: a durable record of every manual change to ``mask/mask.tif``.

Layout (``mask/edits/``):
    log.json                      list of entries, oldest first
    <stamp>_initial.tif           first recorded mask (threshold result + any paint before journaling)
    <stamp>_added.tif             pixels turned ON in that session   (uint8 0/255, zlib)
    <stamp>_removed.tif           pixels turned OFF in that session

The current mask is reproducible as  initial ∪ added₁ − removed₁ ∪ added₂ − removed₂ …  (``replay``), and each entry
records counts, bounding box, the threshold settings in effect (mask.json) and SHA-256 of the mask before/after.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a.astype(bool)).tobytes()).hexdigest()


def _write(p: Path, m: np.ndarray) -> None:
    import tifffile
    tifffile.imwrite(p, (m.astype(bool) * 255).astype(np.uint8), compression="zlib")


def _log_path(mask_path: Path) -> Path:
    return mask_path.parent / "edits" / "log.json"


def load_log(mask_path: Path) -> list:
    p = _log_path(mask_path)
    return json.loads(p.read_text()) if p.exists() else []


def record(mask_path: Path, before: Optional[np.ndarray], after: np.ndarray, note: str = "",
           stamp: Optional[str] = None) -> Optional[dict]:
    """Record one editing session (``before`` None → initial mask).  Returns the log entry, or None if unchanged."""
    d = mask_path.parent / "edits"; d.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    after = after.astype(bool)
    mj = mask_path.parent / "mask.json"
    entry = dict(stamp=stamp, note=note, shape=list(after.shape), sha_after=_sha(after),
                 settings=json.loads(mj.read_text()) if mj.exists() else None)
    if before is None:
        _write(d / f"{stamp}_initial.tif", after)
        entry.update(kind="initial", n_on=int(after.sum()), file=f"{stamp}_initial.tif")
    else:
        before = before.astype(bool)
        add, rem = after & ~before, before & ~after
        if not add.any() and not rem.any():
            return None
        _write(d / f"{stamp}_added.tif", add); _write(d / f"{stamp}_removed.tif", rem)
        rows, cols = np.nonzero(add | rem)
        from scipy.ndimage import label
        entry.update(kind="edit", sha_before=_sha(before), n_added=int(add.sum()), n_removed=int(rem.sum()),
                     n_regions_added=int(label(add)[1]), n_regions_removed=int(label(rem)[1]),
                     bbox_rc=[int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())],
                     files=[f"{stamp}_added.tif", f"{stamp}_removed.tif"])
    log = load_log(mask_path); log.append(entry)
    _log_path(mask_path).write_text(json.dumps(log, indent=1))
    return entry


def replay(mask_path: Path) -> np.ndarray:
    """Rebuild the mask from the journal (initial ∪ added − removed, in order)."""
    import tifffile
    d = mask_path.parent / "edits"; m = None
    for e in load_log(mask_path):
        if e["kind"] == "initial":
            m = tifffile.imread(d / e["file"]) > 0
        else:
            m = (m | (tifffile.imread(d / e["files"][0]) > 0)) & ~(tifffile.imread(d / e["files"][1]) > 0)
    return m
