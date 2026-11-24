"""Read-only access to a frozen plexusflow data release — the interface downstream (modelling) code builds on.

    from plexusflow.release import load_release
    R = load_release("/path/to/PlexusFlow21S_release_v2")
    R.graph            # measured vessel graph (networkx), Q(t) traces re-attached
    R.dc, R.ac         # final DC reconciliation and tile-aligned AC (waveform) field, carved-network edge order
    R.domain           # carved network (incidence, boundary), same edge order as R.dc / R.ac
    R.phase_determined # AC-eligible vessels whose tile time gauge is well determined
    R.tiles            # per-tile heart rate and review status
    R.verify()         # sha256 check against MANIFEST.json

Nothing here writes to the release.  ``schema_version`` (MANIFEST.json) names the layout; this module refuses layouts
it does not know, so a modelling package pinned to schema 2 fails loudly instead of misreading a newer release.

Schema 2 (data-v2, 2026-10-15):
    MANIFEST.json, plexusflow.yaml
    graph/graph_vessels_piv.gpickle, graph/export/{nodes,edges,measurements,tiles}.csv, paths.npz, traces.npz
    fuse/organ_fusion_close.tif | organ_fusion.tif, stitch/positions.json
    results/{dc_reconciled_v2.pkl, ac_field_v2.pkl, reliability_model_v1.json, summary.json, *.csv}
Schema 3 (2026-11-02): + figure inputs; results of the earlier method (ac_field_v7.pkl, ac_estimate_v8.json, validation JSONs).
Schema 4 (2026-11-23, the paper): results/{dc_reconciled_v3.pkl, ac_field_v9.pkl, ac_estimate_v11.json, summary.json, *.csv}
    — mean flow with residual-variance weights and equal left/right inflow; tile clocks from vessels recorded in two tiles; reconciled pulsatile field; distensibility fit.
    The graph holds only RELEASE_FIELDS (`to_release_schema`), in the current field names.
"""
from __future__ import annotations
import csv
import hashlib
import json
import pickle
from functools import cached_property
from pathlib import Path

# Graph fields kept in a release: what the analysis scripts (dc_reconcile, pulse_field, mech/ac_estimate), the paper figures and
# the read-only viewer read.  Established by recording every field lookup while running all of them (2026-11-24); a release
# graph reduced to these fields reproduces results/ and the figures exactly.
RELEASE_FIELDS = {
    "graph": {"tile_f0_piv", "tile_f0_status", "tile_f0_update", "external_traces"},
    "node": {"x", "y", "boundary_type"},
    "edge": {"path", "radius", "length", "open_region", "radial_uncertain", "measurements_piv"},
    "measurement": {"tile_id", "flow_from", "flow_to", "Q_t", "Q_profile", "R_profile_um", "schema_version", "meas_id",
                    "harm", "quality", "snr", "motion"},
    "measurement.harm": {"f0_hz", "mean_Q", "harmonics"},
    "measurement.quality": {"profile_ok", "f", "align", "p"},
    "measurement.snr": {"dc_db", "h1_db"},
    "measurement.motion": {"z_dc", "p_ac"},
}


def to_release_schema(G):
    """Drop every graph field not in RELEASE_FIELDS (in place) and return G.  Expects current field names
    (load through plexusflow.compat)."""
    keep = RELEASE_FIELDS
    for k in [k for k in G.graph if k not in keep["graph"]]:
        del G.graph[k]
    for _, d in G.nodes(data=True):
        for k in [k for k in d if k not in keep["node"]]:
            del d[k]
    for *_, d in G.edges(data=True):
        for k in [k for k in d if k not in keep["edge"]]:
            del d[k]
        ms = []
        for m in d.get("measurements_piv") or []:
            m = {k: v for k, v in m.items() if k in keep["measurement"]}
            for sub in ("harm", "quality", "snr", "motion"):
                if isinstance(m.get(sub), dict):
                    m[sub] = {k: v for k, v in m[sub].items() if k in keep[f"measurement.{sub}"]}
            ms.append(m)
        if "measurements_piv" in d:
            d["measurements_piv"] = ms
    return G


SCHEMA_VERSION = 4          # 4: paper results (dc v3, ac_field_v9, ac_estimate_v11); 3: earlier method (ac_field_v7, ac_estimate_v8); 2: data-v2
SUPPORTED_SCHEMAS = (2, 3, 4)
_DC = {2: "dc_reconciled_v2.pkl", 3: "dc_reconciled_v2.pkl", 4: "dc_reconciled_v3.pkl"}
_AC = {2: "ac_field_v2.pkl", 3: "ac_field_v7.pkl", 4: "ac_field_v9.pkl"}
_FIT = {3: "ac_estimate_v8.json", 4: "ac_estimate_v11.json"}
PHASE_SD_MAX_DEG = 30.0      # tile time gauge bootstrap SD above which phases are not compared across tiles


class ReleaseError(RuntimeError):
    pass


def _schema_of(root: Path, manifest: dict) -> int:
    if "schema_version" in manifest:
        return int(manifest["schema_version"])
    # data-v2 releases predate the explicit field: recognise them by layout
    if (root / "graph" / "export" / "traces.npz").exists() and (root / "results" / "ac_field_v2.pkl").exists():
        return 2
    raise ReleaseError(f"{root}: MANIFEST.json has no schema_version and the layout is not a known release")


class Release:
    """A frozen release folder.  Attributes load lazily and are cached; the folder is never written."""

    def __init__(self, root):
        self.root = Path(root).expanduser()
        mf = self.root / "MANIFEST.json"
        if not mf.exists():
            raise ReleaseError(f"{self.root} is not a plexusflow release (no MANIFEST.json)")
        self.manifest = json.loads(mf.read_text())
        self.schema_version = _schema_of(self.root, self.manifest)
        if self.schema_version not in SUPPORTED_SCHEMAS:
            raise ReleaseError(f"{self.root}: release schema {self.schema_version} not supported by this plexusflow "
                               f"(supports {SUPPORTED_SCHEMAS}); upgrade plexusflow")

    def __repr__(self):
        g = self.manifest.get("graph", {})
        return (f"Release({self.root.name!r}, schema {self.schema_version}, {g.get('nodes', '?')} nodes, "
                f"{g.get('edges', '?')} edges, code {str(self.manifest.get('code_commit', '?'))[:7]})")

    # ---- files
    @property
    def name(self) -> str:
        return self.manifest.get("name", self.root.name)

    @property
    def graph_path(self) -> Path:
        return self.root / "graph" / "graph_vessels_piv.gpickle"

    @property
    def mosaic_path(self) -> Path | None:
        return next((p for p in (self.root / "fuse" / "organ_fusion_close.tif", self.root / "fuse" / "organ_fusion.tif")
                     if p.exists()), None)

    @property
    def positions_path(self) -> Path:
        return self.root / "stitch" / "positions.json"

    def _result(self, name):
        p = self.root / "results" / name
        if not p.exists():
            raise ReleaseError(f"{self.name}: missing results/{name}")
        return p

    # ---- contents
    @cached_property
    def graph(self):
        from .paths import load_graph
        return load_graph(self.graph_path)

    @cached_property
    def dc(self) -> dict:
        return pickle.loads(self._result(_DC[self.schema_version]).read_bytes())

    @cached_property
    def ac(self) -> dict:
        """The pulsatile field on the common clock (schema 4: ac_field_v9, plexusflow.ac.pulse; earlier schemas: ac_field_v7 / v2)."""
        return pickle.loads(self._result(_AC[self.schema_version]).read_bytes())

    @cached_property
    def ac_fit(self) -> dict:
        """The distensibility fit (schema ≥ 3; schema 4: results/ac_estimate_v11.json)."""
        if self.schema_version < 3:
            raise ReleaseError(f"{self.name}: schema {self.schema_version} releases have no distensibility fit")
        return json.loads(self._result(_FIT[self.schema_version]).read_text())

    @cached_property
    def reliability(self) -> dict:
        return json.loads(self._result("reliability_model_v1.json").read_text())

    @cached_property
    def summary(self) -> dict:
        return json.loads(self._result("summary.json").read_text())

    @cached_property
    def tiles(self) -> list[dict]:
        p = self.root / "graph" / "export" / "tiles.csv"
        with open(p, newline="") as f:
            return list(csv.DictReader(f))

    @cached_property
    def domain(self) -> dict:
        """Carved network rebuilt from the graph; checked against the edge order the results were computed on."""
        from .reconcile.network import carved_domain
        dom = carved_domain(self.graph)
        if [tuple(e) for e in dom["edges"]] != [tuple(e) for e in self.dc["edges"]]:
            raise ReleaseError(f"{self.name}: carved network does not match the mean-flow result's edge order")
        return dom

    @property
    def phase_determined(self):
        from .ac.field import phase_determined
        return phase_determined(self.ac, PHASE_SD_MAX_DEG)

    # ---- integrity
    def verify(self, tiers=("core",)) -> list[str]:
        """sha256 of every file of the given tiers against MANIFEST.json; returns the problems (empty = intact)."""
        bad = []
        for rel, meta in self.manifest.get("files", {}).items():
            if meta.get("tier", "core") not in tiers:
                continue
            p = self.root / rel
            if not p.exists():
                bad.append(f"missing {rel}")
                continue
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while b := f.read(1 << 22):
                    h.update(b)
            if h.hexdigest() != meta.get("sha256"):
                bad.append(f"checksum {rel}")
        return bad


def load_release(root) -> Release:
    """Open a release folder (see module docstring).  Raises ReleaseError for unknown or incomplete layouts."""
    return Release(root)
