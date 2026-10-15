"""plexusflow.release: schema handling and (when a release is available) a read-only round trip."""
import json
import os
from pathlib import Path

import pytest

from plexusflow.release import ReleaseError, load_release


def test_rejects_non_release(tmp_path):
    with pytest.raises(ReleaseError, match="not a plexusflow release"):
        load_release(tmp_path)


def test_rejects_unknown_schema(tmp_path):
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(ReleaseError, match="schema 99 not supported"):
        load_release(tmp_path)


def test_rejects_unrecognised_layout(tmp_path):
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"name": "x"}))
    with pytest.raises(ReleaseError, match="no schema_version"):
        load_release(tmp_path)


REL = os.environ.get("PLEXUSFLOW_TEST_RELEASE")


@pytest.mark.skipif(not REL, reason="set PLEXUSFLOW_TEST_RELEASE to an extracted release folder")
def test_real_release_round_trip():
    R = load_release(REL)
    assert R.schema_version == 2
    before = {p: p.stat().st_mtime for p in Path(REL).rglob("*") if p.is_file()}
    dom = R.domain                                  # checks carved edge order against the results
    assert len(dom["edges"]) == len(R.dc["Qp"]) == len(R.ac["elig"])
    assert R.phase_determined.sum() <= R.ac["elig"].sum()
    assert R.tiles and "f0_hz" in R.tiles[0]
    assert R.verify(tiers=("core",)) == []
    assert {p: p.stat().st_mtime for p in Path(REL).rglob("*") if p.is_file()} == before
