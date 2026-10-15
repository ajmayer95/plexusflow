"""Threshold + paint mask editor wrapper.

For now this shells out to PerTileFlow's `scripts/plexus_threshold.py`.
Same porting story as plexusflow.stitch.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..config import StageConfig


def _locate_threshold_script() -> Path:
    env = os.environ.get("PLEXUSFLOW_THRESHOLD_SCRIPT")
    if env:
        return Path(env).expanduser()
    candidates = [
        Path.home() / "Library/CloudStorage/GoogleDrive-amayer@flatironinstitute.org/"
                       "My Drive/Python/Yolk Sac Code/PerTileFlow/scripts/plexus_threshold.py",
        Path(__file__).resolve().parents[4] / "PerTileFlow/scripts/plexus_threshold.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "plexus_threshold.py not found. Set $PLEXUSFLOW_THRESHOLD_SCRIPT.")


def launch_mask_editor(cfg: StageConfig,
                        prob_source: str = "probs_logit",
                        video: str | None = None,
                        invert_prob: bool = False) -> int:
    """Open the threshold+paint editor.

    prob_source ∈ {'probs_logit', 'probs_linear', 'intensity', 'maxmed'} —
    which fused image to threshold.
    """
    cfg.ensure_dirs()
    script = _locate_threshold_script()
    prob_map = {
        "probs_logit":  cfg.fuse_probs_logit_path,
        "probs_linear": cfg.fuse_probs_linear_path,
        "intensity":    cfg.fuse_intensity_path,
        "maxmed":       cfg.fuse_maxmed_path,
    }
    if prob_source not in prob_map:
        raise ValueError(f'unknown prob_source: {prob_source}')
    prob_path = prob_map[prob_source]
    if not prob_path.exists():
        raise FileNotFoundError(
            f'{prob_path} does not exist. Run `plexusflow fuse` first.')

    cmd = [sys.executable, str(script),
           "--prob", str(prob_path),
           "--intensity", str(cfg.fuse_intensity_path
                                if cfg.fuse_intensity_path.exists() else prob_path),
           "--output", str(cfg.mask_path),
           "--init-threshold", str(cfg.mask_default_threshold),
           "--init-low-threshold", str(cfg.mask_default_low),
           "--init-min-size", str(cfg.mask_default_min_size),
           "--init-close-radius", str(cfg.mask_default_close_radius)]
    if cfg.mask_path.exists():
        cmd += ["--load-mask", str(cfg.mask_path)]
    if video:
        cmd += ["--video", str(Path(video).expanduser())]
    if invert_prob:
        cmd += ["--invert-prob"]
    # Optional per-tile video overlay (same StageConfig fields as the
    # graph editor). Prefers videos_realtime/ over raw/.
    if cfg.positions_path.exists():
        video_dir = None
        if (cfg.videos_realtime_dir.exists()
                and any(cfg.videos_realtime_dir.iterdir())):
            video_dir = cfg.videos_realtime_dir
        elif cfg.raw_video_dir.exists():
            video_dir = cfg.raw_video_dir
        if video_dir is not None:
            cmd += ["--tile-positions", str(cfg.positions_path),
                     "--video-dir",       str(video_dir),
                     "--video-pattern",   cfg.raw_video_dir_pattern]

    # Snapshot the mask before editing so the session's changes are journaled (mask/edits/).
    from . import journal
    before = None
    if cfg.mask_path.exists():
        import tifffile
        before = tifffile.imread(cfg.mask_path) > 0
        if not journal.load_log(cfg.mask_path):
            journal.record(cfg.mask_path, None, before, note="mask before journaling began")
    print(f'[mask] launching threshold editor on {prob_path.name} '
          f'(source={prob_source})')
    rc = subprocess.call(cmd)
    if rc == 0 and cfg.mask_path.exists():
        import tifffile
        after = tifffile.imread(cfg.mask_path) > 0
        e = journal.record(cfg.mask_path, before, after,
                           note="mask editor session" if before is not None else "initial threshold + paint")
        if e is not None:
            print(f"[mask] journaled: {e.get('n_added', e.get('n_on'))} px added, {e.get('n_removed', 0)} removed "
                  f"→ {cfg.mask_path.parent / 'edits'}")
        cfg.write_provenance("mask", extra={
            "prob_source": prob_source, "prob_path": str(prob_path),
            "video": video, "invert_prob": invert_prob,
        })
    return rc
