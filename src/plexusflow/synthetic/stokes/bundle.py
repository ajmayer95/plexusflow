"""The benchmark bundle: everything the known-truth benchmark reads, so it reruns without raw videos or a mesher.

    <stage>_benchmark_v1/
      MANIFEST.json                      schema, code commit, sha256 per file
      backgrounds/<stage>_tile<t>.npz    per real tile, full resolution: bg<nf> (per-pixel 10th percentile of the first nf
                                         raw frames), sd<nf> (per-pixel temporal SD), gain, read (photon-transfer fit)
      junctions/<node>.{npz,pkl}         Stokes basis solutions on real Y-junctions (mesh p, t; uA, uB) + branch geometry
      vessels/<name>.{npz,pkl}           Stokes solution on controlled single vessels (mesh p, t; u) + centreline metadata

Built by scripts/benchmark/make_bundle.py from a stage folder + raw videos (+ gmsh / scikit-fem for the geometry).
Cropping a stored background equals computing the percentile / SD on the cropped raw clip (per-pixel statistics), so
benchmark runs from the bundle reproduce runs from raw data exactly.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


class Background:
    """One real tile: static background and temporal SD over the first nf frames, and the camera noise model."""

    def __init__(self, bg, sd, gain, read, stage, tile, nf):
        # numpy float64, not Python float: under NumPy 2 promotion a float32 / Python float stays float32, which would change
        # the noise model in the last bits relative to runs from raw data
        self.bg, self.sd, self.gain, self.read = bg, sd, np.float64(gain), np.float64(read)
        self.stage, self.tile, self.nf = stage, tile, nf

    def crop(self, r0, r1, c0, c1):
        """(bg, sd) on rows r0:r1, cols c0:c1 (numpy slicing semantics, as when cropping the raw frames)."""
        return self.bg[r0:r1, c0:c1], self.sd[r0:r1, c0:c1]


def tile_noise(frames_u8_iter):
    """Photon-transfer fit var = mean/g + r² from the lower envelope (10th percentile of variance per brightness bin) of
    200 frames subsampled ::3 in space.  Returns (gain g per count, read noise r in counts)."""
    full = np.stack([f[::3, ::3] for f in frames_u8_iter]).astype(np.float32)
    mu, var = full.mean(0).ravel(), full.var(0).ravel(); bins = np.quantile(mu, np.linspace(0.02, 0.9, 15)); xb, yb = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        s = (mu >= a) & (mu < b)
        if s.sum() > 50:
            xb.append(mu[s].mean()); yb.append(np.percentile(var[s], 10))
    (gi, r2), *_ = np.linalg.lstsq(np.column_stack([xb, np.ones(len(xb))]), yb, rcond=None)
    return 1 / max(gi, 1e-4), np.sqrt(max(r2, 0))


def background_from_raw(files, nfs=(400,)):
    """Background arrays for a tile from its raw frame files (sorted).  Returns a dict ready for np.savez_compressed."""
    import tifffile
    out = {}
    gain, read = tile_noise(tifffile.imread(f) for f in files[:200])
    out["gain"], out["read"] = np.float64(gain), np.float64(read)
    clip = np.stack([tifffile.imread(f) for f in files[:max(nfs)]]).astype(np.float32)
    for nf in nfs:
        out[f"bg{nf}"] = np.percentile(clip[:nf], 10, axis=0)
        out[f"sd{nf}"] = clip[:nf].std(0)
    return out


class Bundle:
    def __init__(self, root):
        self.root = Path(root).expanduser()
        mf = self.root / "MANIFEST.json"
        self.manifest = json.loads(mf.read_text()) if mf.exists() else {}
        sv = self.manifest.get("schema_version", SCHEMA_VERSION)
        if sv != SCHEMA_VERSION:
            raise RuntimeError(f"{self.root}: benchmark bundle schema {sv}, this code reads {SCHEMA_VERSION}")
        self._bg = {}

    def __repr__(self):
        return (f"Bundle({self.root.name!r}: {len(self.junctions())} junctions, {len(self.vessels())} vessels, "
                f"{len(list((self.root / 'backgrounds').glob('*.npz')))} backgrounds)")

    def background(self, stage: str, tile: int, nf: int) -> Background:
        key = (stage, tile, nf)
        if key not in self._bg:
            p = self.root / "backgrounds" / f"{stage}_tile{tile}.npz"
            if not p.exists():
                raise FileNotFoundError(f"no background for {stage} tile {tile} in {self.root}")
            Z = np.load(p)
            if f"bg{nf}" not in Z.files:
                raise KeyError(f"{p.name} has no {nf}-frame background (has {[k for k in Z.files if k.startswith('bg')]})")
            self._bg[key] = Background(Z[f"bg{nf}"], Z[f"sd{nf}"], Z["gain"], Z["read"], stage, tile, nf)
        return self._bg[key]

    def junctions(self) -> list[int]:
        return sorted(int(p.stem) for p in (self.root / "junctions").glob("*.pkl"))

    def junction(self, node: int):
        """(meta, arrays): meta = dict(node, tile, branches, node_xy); arrays p, t, uA, uB."""
        d = self.root / "junctions"
        return pickle.loads((d / f"{node}.pkl").read_bytes()), np.load(d / f"{node}.npz")

    def vessels(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "vessels").glob("*.pkl"))

    def vessel(self, name: str):
        """(meta, arrays): meta = dict(kind, a0, eps, taper, P, meas, s, a_meas, caps); arrays p, t, u."""
        d = self.root / "vessels"
        return pickle.loads((d / f"{name}.pkl").read_bytes()), np.load(d / f"{name}.npz")
