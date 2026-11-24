"""PLOS Computational Biology figure style shared by the paper figure scripts.

Journal limits: width ≤ 7.5 in, height ≤ 8.75 in, text 8–12 pt Arial, TIFF (LZW, flattened RGB, no alpha) at 300–600 dpi,
≤ 10 MB per file.  House style on top of that: axis labels and titles 9 pt, ticks and legends 8 pt, every other text ≥ 8 pt,
panel letters 12 pt bold, axis and tick lines 0.8 pt.

    import plos_style as ps
    ps.apply()                      # before building the figure
    ...
    ps.save(fig, OUT_DIR / "fig5")  # finalize sizes, then writes fig5.png / .pdf / .tif and checks the limits
"""
from __future__ import annotations
import io
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

MAX_W, MAX_H, MAX_MB = 7.5, 8.75, 10.0
LABEL, TICK, MIN, LETTER, LW = 9.0, 8.0, 8.0, 12.0, 0.8


def apply():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": LABEL, "axes.titlesize": LABEL, "axes.labelsize": LABEL,
        "xtick.labelsize": TICK, "ytick.labelsize": TICK, "legend.fontsize": TICK,
        "mathtext.fontset": "cm",                    # all mathematics in LaTeX Computer Modern (Alex, 2026-11-19); text stays Arial
        "axes.linewidth": LW, "xtick.major.width": LW, "ytick.major.width": LW, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.2, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.transparent": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    try:
        plt.rcParams["mathtext.default"] = "normal"      # LaTeX behaviour: italic letters, upright digits (matplotlib ≥ 3.10)
    except ValueError:                                   # older matplotlib: italic throughout; the pinned version renders as published
        import warnings
        warnings.warn(f"matplotlib {matplotlib.__version__} has no mathtext.default 'normal'; using 'it'. "
                      "Install from requirements-lock.txt to match the published figures.", stacklevel=2)
        plt.rcParams["mathtext.default"] = "it"


def _is_letter(t):
    s = t.get_text().strip()
    return len(s) == 1 and s.isalpha() and s.islower() and t.get_fontweight() in ("bold", 700, "heavy")


def finalize(fig, letter=LETTER):
    """All mathematics is LaTeX Computer Modern (rcParams can't be overridden per script: checked here).
    Raise text to the house sizes (never shrinks anything except panel letters, which are set to exactly `letter`)."""
    for ax in fig.axes:
        for lab in (ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title):
            if lab.get_text() and lab.get_fontsize() < LABEL:
                lab.set_fontsize(LABEL)
        ax.tick_params(which="major", width=LW)
        for sp in ax.spines.values():
            if sp.get_visible() and sp.get_linewidth() < LW:
                sp.set_linewidth(LW)
    if plt.rcParams["mathtext.fontset"] != "cm" or plt.rcParams["mathtext.default"] == "regular":
        raise RuntimeError("figure math must be LaTeX Computer Modern: do not override mathtext.fontset / mathtext.default")
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_text():
            continue
        if _is_letter(t):
            t.set_fontsize(letter)
        elif t.get_fontsize() < MIN:
            t.set_fontsize(MIN)


def save(fig, stem, dpi=300, pdf=True, png=True, finalize_fig=True):
    """finalize → stem.png (300 dpi preview), stem.pdf (vector working copy), stem.tif (submission: RGB, LZW, `dpi`)."""
    stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    if finalize_fig:                                          # False: the script has already set every text size itself
        finalize(fig)
    w, h = fig.get_size_inches()
    if w > MAX_W + 1e-6 or h > MAX_H + 1e-6:
        print(f"[plos] WARNING {stem.name}: {w:.2f} × {h:.2f} in exceeds {MAX_W} × {MAX_H} in")
    if png:
        fig.savefig(f"{stem}.png", dpi=300, facecolor="white")
    if pdf:
        fig.savefig(f"{stem}.pdf", facecolor="white")
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, facecolor="white"); buf.seek(0)
    im = Image.open(buf).convert("RGB")                       # flatten: no alpha channel
    im.save(f"{stem}.tif", compression="tiff_lzw", dpi=(dpi, dpi))
    mb = Path(f"{stem}.tif").stat().st_size / 1e6
    print(f"[plos] {stem.name}: {w:.2f} × {h:.2f} in, TIFF {im.size[0]}×{im.size[1]} px @ {dpi} dpi, {mb:.1f} MB"
          + ("  — WARNING > 10 MB" if mb > MAX_MB else ""))
