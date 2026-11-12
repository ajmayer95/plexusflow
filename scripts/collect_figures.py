"""Copy the current main-text figures (Figs 1–9) into one folder for sharing / submission.

Reads $PLEXUSFLOW_STAGE/figures/figN/figN.{tif,pdf,png} (written by the make_figure*.py scripts, see FIGURES.md) and copies
them to $PLEXUSFLOW_STAGE/figures/latest/ as Fig1.tif, Fig1.pdf, Fig1.png, …  Copies, not links (the stage folder lives on
Google Drive).  Files in latest/ that no longer correspond to a figure are left alone.  Re-run after any figure re-render.
"""
from __future__ import annotations
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plexusflow.paths as _paths

N_FIGS = 9
EXTS = ("tif", "pdf", "png")


def main():
    src_root = _paths.FIGURES; out = src_root / "latest"; out.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for k in range(1, N_FIGS + 1):
        for ext in EXTS:
            src = src_root / f"fig{k}" / f"fig{k}.{ext}"
            if not src.exists():
                missing.append(str(src.relative_to(src_root))); continue
            dst = out / f"Fig{k}.{ext}"
            shutil.copy2(src, dst)
            if ext == "tif":
                rows.append(f"Fig{k}\t{datetime.fromtimestamp(src.stat().st_mtime):%Y-%m-%d %H:%M}\tfigures/fig{k}/fig{k}.*")
    (out / "README.txt").write_text("Latest main-text figures, copied by scripts/collect_figures.py — do not edit here;\n"
                                    "re-render with scripts/make_figure*.py (see FIGURES.md) and re-run the collector.\n\n"
                                    "figure\trendered\tsource\n" + "\n".join(rows) + "\n")
    print(f"[collect] {len(rows)} figures → {out}")
    for r in rows:
        print("  " + r.replace("\t", "  "))
    if missing:
        print("[collect] missing: " + ", ".join(missing))


if __name__ == "__main__":
    main()
