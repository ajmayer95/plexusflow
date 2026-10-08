"""Graph editor wrapper. Thin over `net3.cli edit-mask` (which does
vectorize + edit in one napari session) or `net3.cli edit` (edit an
existing graph, mask as reference)."""
from __future__ import annotations

import subprocess
import sys

from ..config import StageConfig


def launch_editor(cfg: StageConfig, from_mask: bool = False) -> int:
    """Launch the net3 editor.

    from_mask=True  → net3 edit-mask (vectorize on the fly from mask.tif)
    from_mask=False → net3 edit (opens the existing graph.gpickle)
    """
    if from_mask:
        if not cfg.mask_path.exists():
            raise FileNotFoundError(f'no mask at {cfg.mask_path}')
        cmd = [sys.executable, "-m", "net3.cli", "edit-mask",
               str(cfg.mask_path),
               "--save", str(cfg.graph_path)]
        if cfg.fuse_intensity_path.exists():
            cmd += ["--image", str(cfg.fuse_intensity_path)]
        if cfg.fuse_probs_logit_path.exists():
            cmd += ["--probs", str(cfg.fuse_probs_logit_path)]
        # Forward the per-tile video overlay too, if available.
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
    else:
        if not cfg.graph_path.exists():
            raise FileNotFoundError(f'no graph at {cfg.graph_path}. '
                                    'Run `plexusflow vectorize` first or use --from-mask.')
        cmd = [sys.executable, "-m", "net3.cli", "edit",
               str(cfg.graph_path)]
        if cfg.mask_path.exists():
            cmd += ["-m", str(cfg.mask_path)]
    # Optional per-tile video overlay — reuses the same config fields the
    # viewer uses. Prefer videos_realtime/ (fast playback) over raw/ .
    if cfg.positions_path.exists():
        video_dir = None
        if cfg.videos_realtime_dir.exists() and any(cfg.videos_realtime_dir.iterdir()):
            video_dir = cfg.videos_realtime_dir
        elif cfg.raw_video_dir.exists():
            video_dir = cfg.raw_video_dir
        if video_dir is not None:
            cmd += ["--tile-positions", str(cfg.positions_path),
                     "--video-dir",       str(video_dir),
                     "--video-pattern",   cfg.raw_video_dir_pattern]
    return subprocess.call(cmd)
