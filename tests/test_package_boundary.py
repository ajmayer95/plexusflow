"""Package boundary: plexusflow (measurement → conservative field) never depends on plexusmech (modelling) or on
its own legacy normalization, except for the explicitly optional hooks below."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
# plexusflow files allowed to import plexusmech lazily (inside a function): the workbench's Simulate tab plugin
MECH_HOOKS = {"plexusflow/viewer/app.py", "plexusflow/viewer/simulate.py",   # Simulate tab (lazy, function-level)
              "plexusflow/viewer/model.py"}                                # Model source (lazy; off without plexusmech)
# plexusflow files allowed to import plexusflow.legacy lazily: the `legacy-normalize` CLI verb
LEGACY_HOOKS = {"plexusflow/cli.py"}


def _imports(path):
    """(module name, at_module_level) for every import in a file; relative imports resolved against the package."""
    tree = ast.parse(path.read_text())
    pkg = list(path.relative_to(SRC).with_suffix("").parts[:-1])
    top = {id(n) for n in tree.body}
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name, id(node) in top) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = pkg[: len(pkg) - node.level + 1] if node.level else []
            mod = ".".join(base + ([node.module] if node.module else []))
            out.append((mod, id(node) in top))
    return out


def _files(pkg):
    return sorted(p for p in (SRC / pkg).rglob("*.py"))


def test_plexusflow_does_not_import_plexusmech():
    bad = []
    for f in _files("plexusflow"):
        rel = str(f.relative_to(SRC))
        for mod, top in _imports(f):
            if mod.split(".")[0] == "plexusmech" and (rel not in MECH_HOOKS or top):
                bad.append(f"{rel}: {mod}")
    assert not bad, "plexusflow must not depend on plexusmech:\n" + "\n".join(bad)


def test_pipeline_does_not_use_legacy_normalization():
    bad = []
    for f in _files("plexusflow"):
        rel = str(f.relative_to(SRC))
        if rel.startswith("plexusflow/legacy/"):
            continue
        for mod, top in _imports(f):
            if mod.startswith("plexusflow.legacy") and (rel not in LEGACY_HOOKS or top):
                bad.append(f"{rel}: {mod}")
    assert not bad, "only the legacy-normalize CLI verb may use plexusflow.legacy:\n" + "\n".join(bad)


def test_plexusmech_does_not_use_legacy_normalization():
    bad = [f"{f.relative_to(SRC)}: {m}" for f in _files("plexusmech") for m, _ in _imports(f) if m.startswith("plexusflow.legacy")]
    assert not bad, "\n".join(bad)


def test_core_modules_import_without_plexusmech(monkeypatch):
    """The reconstruction API imports with plexusmech hidden."""
    import importlib, sys
    monkeypatch.setitem(sys.modules, "plexusmech", None)
    for m in ("plexusflow.reconcile", "plexusflow.reconcile.carve", "plexusflow.ac.field", "plexusflow.release",
              "plexusflow.paths", "plexusflow.cli"):
        sys.modules.pop(m, None)
        importlib.import_module(m)
