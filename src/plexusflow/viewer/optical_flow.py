"""Fresh optical flow on a vessel, Kloosterman slices, PIV comparison and OF diagnostics.

Mixin of plexusflow.viewer.app.ReadOnlyMosaicViewer (methods moved verbatim from app.py)."""
from __future__ import annotations

from ._common import (  # noqa: F401
    _note_suppressed,
    TILE_RAW_HEIGHT,
    TILE_RAW_WIDTH,
    np,
)


class OpticalFlowMixin:

    # ── Optical-flow re-analysis (read-only) ──────────────────────────
    def _of_tile_geometry(self):
        """Parse `tile_positions` into TileInfo + reference affine, applying
        the SAME re-referencing the batch does (batch_analyze_mosaic_piv lines
        912-926), cached.  Returns (tiles_by_vid, affine_sx, affine_sy) and
        stores the batch tile offsets on self.  This lets fresh OF reuse the
        batch's `mosaic_to_tile_coords` / `mosaic_radius_to_tile` so a
        recompute reproduces the stored Q(t) exactly."""
        if getattr(self, '_of_tinfo', None) is not None:
            return self._of_tinfo, self._of_affine_sx, self._of_affine_sy
        from ..cli.batch_analyze_mosaic import (
            parse_tile_positions, _get_reference_affine)
        import json as _json
        tp = str(self.tile_positions_path)
        tiles = parse_tile_positions(tp)
        affine_sy, affine_sx = _get_reference_affine(tp)
        jd = _json.load(open(tp)); ref = int(jd.get('reference_vid', 1))
        if abs(affine_sy - 1) > 1e-6 or abs(affine_sx - 1) > 1e-6:
            rtd = jd['tiles'][str(ref)]
            rty = rtd.get('translate_y', 0.0); rtx = rtd.get('translate_x', 0.0)
            for t in tiles.values():
                oty = t._translate_y or 0.0; otx = t._translate_x or 0.0
                t._translate_y = (oty - rty) / affine_sy
                t._translate_x = (otx - rtx) / affine_sx
                t.tile_width /= affine_sx; t.tile_height /= affine_sy
        self._of_tile_offset_x = min(t.top_left_x for t in tiles.values())
        self._of_tile_offset_y = min(t.top_left_y for t in tiles.values())
        self._of_tinfo = tiles
        self._of_affine_sx = affine_sx; self._of_affine_sy = affine_sy
        return tiles, affine_sx, affine_sy


    def _show_kloosterman_slices(self):
        """Per-slice Kloosterman diagnostic — R, V_CL, Q along the vessel
        arc, with median lines and comparison to Q_direct / Q_pois_free.
        Shows how the median-of-slices Q_kl is derived and which slices
        got gated out (R constraint / low-signal)."""
        import matplotlib.pyplot as plt
        from .mosaic import show_figure
        data = getattr(self, '_last_of_pipeline', None)
        if not data:
            self._info_box.setPlainText(
                "Run 'Compute fresh PIV/OF' on an edge first.")
            return
        res = data['res']
        v_conv = data['v_conv']; q_conv = data['q_conv']
        r_tile_px = float(data.get('r_tile', np.nan))
        px_um = q_conv * (0.0 if q_conv == 0 else 1)  # will overwrite below
        # Prefer analyze constants for the µm scale (px→µm on R axis)
        try:
            from ..analyze.constants import PX_SIZE_UM
            _px_um = float(PX_SIZE_UM)
        except Exception:
            _px_um = 1.7
        Q_slice_px = np.asarray(res.get('Q_kloosterman_per_slice', []),
                                 dtype=float)
        R_slice_px = np.asarray(res.get('R_kloosterman_per_slice', []),
                                 dtype=float)
        if len(Q_slice_px) == 0:
            self._info_box.setPlainText(
                "No Kloosterman per-slice data in the OF result.\n"
                "(Batch stores medians; re-run 'Compute fresh PIV/OF' "
                "which populates the per-slice arrays.)")
            return
        # Physical units.  Our stored `Q_kloosterman_per_slice` uses
        # `use_r_prior_area=True` — Q_slice = ½·V_CL·π·R_prior² — i.e.
        # V_CL from the Poiseuille fit but AREA fixed to the mask radius.
        # The Kloosterman 2014 paper convention uses the *fitted* R_slice
        # for the area too (Q_slice = ½·V_CL·π·R_fit²), which is what
        # ``Q_slice_paper_nLs`` reproduces here — rescale by (R_fit/R_prior)².
        Q_slice_ours_nLs = Q_slice_px * q_conv        # our default (R_prior area)
        R_slice_um       = R_slice_px * _px_um        # µm per slice (fit)
        with np.errstate(invalid='ignore', divide='ignore'):
            area_ratio = np.where(
                np.isfinite(R_slice_px) & (r_tile_px > 0),
                (R_slice_px / r_tile_px) ** 2, np.nan)
        Q_slice_paper_nLs = Q_slice_ours_nLs * area_ratio
        # V_CL is independent of the area convention — same for both.
        with np.errstate(invalid='ignore', divide='ignore'):
            V_slice_ums = np.where(
                np.isfinite(R_slice_um) & (R_slice_um > 0),
                2.0 * (Q_slice_ours_nLs * 1e6) / (np.pi * (r_tile_px * _px_um) ** 2),
                np.nan)
        arc = np.arange(len(Q_slice_px), dtype=float)
        valid = np.isfinite(Q_slice_px) & (Q_slice_px != 0)
        Q_med_ours  = float(np.nanmedian(Q_slice_ours_nLs[valid]))  if valid.any() else np.nan
        Q_med_paper = float(np.nanmedian(Q_slice_paper_nLs[valid])) if valid.any() else np.nan
        R_med = float(np.nanmedian(R_slice_um[valid])) if valid.any() else np.nan
        V_med = float(np.nanmedian(V_slice_ums[valid])) if valid.any() else np.nan
        # Prior R (from graph) + free-fit R for context lines
        R_prior_um = r_tile_px * _px_um
        R_free_um  = float(res.get('R_fit_free', np.nan)) * _px_um
        V_free_ums = float(res.get('v0_free', np.nan)) * v_conv
        V_fix_ums  = float(res.get('v0_fixed', np.nan)) * v_conv
        Q_direct_nLs = float(V_fix_ums / 1e6) * np.pi * R_prior_um ** 2 / 2.0
        Q_pois_free_nLs = (float(V_free_ums / 1e6)
                            * np.pi * R_free_um ** 2 / 2.0
                            if np.isfinite(R_free_um) else np.nan)

        # ── (0,0) Vessel with per-slice R half-widths ───────────────────
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(15, 11), dpi=110)
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.28,
                       height_ratios=[1.0, 0.85])
        fig.suptitle(f"Kloosterman per-slice — {data.get('title','')}",
                      fontsize=11)

        ax = fig.add_subplot(gs[0, :])
        vfield = res.get('vfield_mean')
        cl_used = res.get('cl_used')     # ROI-local (row, col)
        if vfield is not None and cl_used is not None and len(cl_used) >= 2:
            roi_h, roi_w = vfield.shape[:2]
            # Grayscale background = mean intensity over the ROI (from the
            # frames the OF ran on).  Gives full vessel context so parts
            # outside the tight vessel mask are still visible instead of
            # showing black.
            frames = data.get('frames')
            r0_off, c0_off = res.get('roi_offset', (0, 0))
            bg = None
            if frames is not None:
                fr = np.asarray(frames)
                if fr.ndim == 3 and fr.shape[0] >= 1:
                    mean_frame = fr.mean(axis=0)
                    r1_off = min(mean_frame.shape[0], r0_off + roi_h)
                    c1_off = min(mean_frame.shape[1], c0_off + roi_w)
                    bg = mean_frame[r0_off:r1_off, c0_off:c1_off]
                    lo, hi = np.percentile(bg, (1, 99))
                    if hi > lo:
                        ax.imshow(bg, cmap='gray',
                                   vmin=lo, vmax=hi, alpha=1.0)
            speed_ums = np.sqrt(vfield[..., 0] ** 2 + vfield[..., 1] ** 2) * v_conv
            # Overlay the OF magnitude with alpha where sampled.  Cells
            # outside the vessel mask are transparent so the grayscale
            # intensity below shows through.
            masked_speed = np.ma.masked_where(speed_ums <= 0, speed_ums)
            im = ax.imshow(masked_speed, cmap='hot',
                            vmin=0, vmax=float(np.percentile(speed_ums, 99)),
                            alpha=0.85 if bg is not None else 1.0)
            plt.colorbar(im, ax=ax, label='|v| (µm/s)', shrink=0.75)
            cl = np.asarray(cl_used, dtype=float)
            # Plot the EXACT centerline the analysis used (cl_used).  The
            # refinement now smooths the per-slice offsets
            # (ridge_offset_smooth_sigma), so no display-only smoothing is
            # needed and the drawn curve == the curve Q was computed on.
            cl_smooth = cl.copy()
            ax.plot(cl_smooth[:, 1], cl_smooth[:, 0], color='cyan', lw=1.5,
                     alpha=0.9,
                     label='CL (analysis centerline); segment half-widths = R_fit')
            # Local tangents from finite differences on smoothed CL.
            t = np.zeros_like(cl_smooth)
            t[1:-1] = cl_smooth[2:] - cl_smooth[:-2]
            t[0]    = cl_smooth[1] - cl_smooth[0]
            t[-1]   = cl_smooth[-1] - cl_smooth[-2]
            norm_len = np.hypot(t[:, 0], t[:, 1])
            norm_len[norm_len < 1e-9] = 1.0
            t /= norm_len[:, None]
            n = np.column_stack([t[:, 1], -t[:, 0]])   # perpendicular
            # Decimate slices for display so adjacent perpendiculars don't
            # visually merge into a mess.  Target spacing = R_prior along
            # the arc → visible neighbours don't overlap by more than a
            # slice-width.  Slices in between are shown as small ticks so
            # you can still see coverage.
            N = min(len(cl_smooth), len(R_slice_px))
            arc_len = np.concatenate([[0.0], np.cumsum(np.hypot(
                np.diff(cl_smooth[:, 0]), np.diff(cl_smooth[:, 1])))])
            step_target = max(r_tile_px * 1.0, 4.0)
            show_full = np.zeros(N, dtype=bool)
            last_arc = -np.inf
            for i in range(N):
                if arc_len[i] - last_arc >= step_target:
                    show_full[i] = True
                    last_arc = arc_len[i]
            # ticks: everything not shown-full
            for i in range(N):
                r_use = R_slice_px[i] if np.isfinite(R_slice_px[i]) \
                        else r_tile_px
                if not np.isfinite(r_use) or r_use <= 0:
                    continue
                colour = 'lime' if valid[i] else '#888888'
                if show_full[i]:
                    p1 = cl_smooth[i] - r_use * n[i]
                    p2 = cl_smooth[i] + r_use * n[i]
                    ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color=colour,
                             lw=1.6, alpha=0.9)
                else:
                    p1 = cl_smooth[i] - 2.0 * n[i]
                    p2 = cl_smooth[i] + 2.0 * n[i]
                    ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color=colour,
                             lw=0.6, alpha=0.55)
            n_shown = int(show_full.sum())
            hint = (f'  ·  showing {n_shown}/{N} slice half-widths '
                    f'(spacing ≈ R_prior); the rest as tick marks')
            ax.set_title(
                f'vessel with per-slice fit half-widths — '
                f'{int(valid.sum())}/{N} slices accepted '
                f'(lime = accepted, gray = rejected)  '
                f'R_prior = {R_prior_um:.1f} µm{hint}')
            ax.legend(fontsize=8, loc='upper right',
                       facecolor='black', edgecolor='none',
                       labelcolor='white', framealpha=0.55)
            # Rotate into the mosaic frame (same as the OF-pipeline panel)
            # so this matches the main viewer's orientation.
            try:
                _tid = int(self._selected_tile_id)
                _L = self._of_geom.tiles[_tid].A_full[:2, :2]
                import matplotlib.transforms as _mtx
                _aff = _mtx.Affine2D.from_values(
                    _L[1, 1], _L[0, 1], _L[1, 0], _L[0, 0], 0.0, 0.0)
                for _art in list(ax.images) + list(ax.lines) + list(ax.collections):
                    _art.set_transform(_aff + ax.transData)
                _cor = np.array([[0, 0], [roi_w, 0], [0, roi_h], [roi_w, roi_h]], float)
                _xt = _L[1, 1] * _cor[:, 0] + _L[1, 0] * _cor[:, 1]
                _yt = _L[0, 1] * _cor[:, 0] + _L[0, 0] * _cor[:, 1]
                _pad = 0.05 * max(np.ptp(_xt), np.ptp(_yt), 1.0)
                ax.set_xlim(_xt.min() - _pad, _xt.max() + _pad)
                ax.set_ylim(_yt.max() + _pad, _yt.min() - _pad)
                ax.set_aspect('equal')
            except Exception as _e:
                print(f"  (KL panel orient skipped: {_e})")
            ax.set_xlabel('col (ROI px)'); ax.set_ylabel('row (ROI px)')
        else:
            ax.text(0.5, 0.5,
                     'no velocity field / centerline available',
                     transform=ax.transAxes, ha='center', va='center',
                     color='gray')
            ax.set_xticks([]); ax.set_yticks([])

        # ── (1,0) R per slice ─────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(arc[valid], R_slice_um[valid], 'ko', ms=4,
                 label=f'per-slice R  (n={int(valid.sum())})')
        ax.axhline(R_med, color='green', ls='--', lw=1.4,
                    label=f'median R = {R_med:.1f} µm')
        ax.axhline(R_prior_um, color='blue', ls=':', lw=1.2,
                    label=f'R_prior (graph) = {R_prior_um:.1f} µm')
        if np.isfinite(R_free_um):
            ax.axhline(R_free_um, color='red', ls=':', lw=1.2,
                        label=f'R_free = {R_free_um:.1f} µm')
        ax.axhspan(0.2 * R_prior_um, 3.0 * R_prior_um,
                    color='yellow', alpha=0.06,
                    label='gate ×[0.2, 3.0] R_prior')
        ax.set_xlabel('slice index'); ax.set_ylabel('R (µm)')
        ax.set_title('per-slice radius'); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='best')

        # ── (1,1) V_CL per slice ─────────────────────────────────────
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(arc[valid], V_slice_ums[valid], 'ko', ms=4,
                 label='per-slice V_CL')
        if np.isfinite(V_med):
            ax.axhline(V_med, color='green', ls='--', lw=1.4,
                        label=f'median V_CL = {V_med:.0f} µm/s')
        if np.isfinite(V_fix_ums):
            ax.axhline(V_fix_ums, color='blue', ls=':', lw=1.2,
                        label=f'V0 (fixed R fit) = {V_fix_ums:.0f}')
        if np.isfinite(V_free_ums):
            ax.axhline(V_free_ums, color='red', ls=':', lw=1.2,
                        label=f'V0 (free R fit) = {V_free_ums:.0f}')
        ax.set_xlabel('slice index'); ax.set_ylabel('V_CL (µm/s)')
        ax.set_title('per-slice centerline velocity'); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='best')

        # ── (1,2) Q per slice — ours (R_prior area) vs paper (R_fit area) ─
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(arc[valid], Q_slice_ours_nLs[valid], 'o',
                 color='#1f77b4', ms=4,
                 label=f'ours: Q = ½·V_CL·π·R_prior²  (n={int(valid.sum())})')
        ax.plot(arc[valid], Q_slice_paper_nLs[valid], 's',
                 color='#d62728', ms=4, alpha=0.85,
                 label='paper: Q = ½·V_CL·π·R_fit²')
        if np.isfinite(Q_med_ours):
            ax.axhline(Q_med_ours, color='#1f77b4', ls='--', lw=1.6,
                        label=f'median (ours, current default) = {Q_med_ours:.3f} nL/s')
        if np.isfinite(Q_med_paper):
            ax.axhline(Q_med_paper, color='#d62728', ls='--', lw=1.6,
                        label=f'median (Kloosterman 2014) = {Q_med_paper:.3f} nL/s')
        if np.isfinite(Q_direct_nLs):
            ax.axhline(Q_direct_nLs, color='gray', ls=':', lw=1.0,
                        label=f'Q_direct = {Q_direct_nLs:.3f}')
        if np.isfinite(Q_pois_free_nLs):
            ax.axhline(Q_pois_free_nLs, color='black', ls=':', lw=1.0,
                        label=f'Q_pois_free = {Q_pois_free_nLs:.3f}')
        ax.set_xlabel('slice index'); ax.set_ylabel('Q (nL/s)')
        ax.set_title('per-slice Q — ours vs paper'); ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc='best')
        show_figure(fig, position='top-right')


    def _render_of_pipeline(self, data):
        """Browsable SEQUENCE of figures walking the entire OF pipeline, so
        every processing step is visible on its own page:

          Fig 1  Input & preprocessing  — ROI selection, background
                 subtraction, mean flow field.
          Fig 2  Centerline refinement  — coarse rigid shift, per-slice
                 parabola snap, offset smoothing, final CL.
          Fig 3  Radii & cross-sections — vessel mask + tangent, and the
                 SEGMENTED (mask) radius vs the INFERRED (fit) radius drawn
                 directly on the video crop AND the speed heatmap, with a
                 perpendicular velocity-arrow fan at several cross-sections
                 (arrow length = the fitted Poiseuille profile).  Plus the
                 radial profile with all three fits.
          Fig 4  Kloosterman deep-dive  — per-slice R / V_CL / Q, the
                 median-band gate, the reject-reason breakdown, the vessel
                 with per-slice half-widths, and how (Q_char, V_CL_char,
                 R_char) are derived.
          Fig 5  Q estimators & temporal — estimator comparison, cycle Q(t),
                 FFT/SNR, and the overall verdict.

        Driven entirely by the ONE `unified` result dict (+ its `diag`
        block) — what you see here is exactly what produced the reported Q.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        from scipy.ndimage import uniform_filter1d
        from .mosaic import show_figure
        try:
            from ..analyze.centerline import centerline_geometry
        except Exception:
            centerline_geometry = None

        res = data['res']
        diag = res.get('diag') or {}
        v_conv = float(data['v_conv']); q_conv = float(data['q_conv'])
        r_tile = float(data.get('r_tile', np.nan))            # R_prior (px)
        frame_dt_s = float(data.get('frame_dt_s', 1.0 / 250))
        f0_hz = float(data.get('f0_hz') or 2.5)
        best_key = data.get('best_key', 'kl')
        u, v = data.get('edge', (0, 0))
        tile_id = data.get('tile_id')
        px_um = v_conv * frame_dt_s                            # effective µm/px
        R_prior_um = r_tile * px_um

        r0, c0 = diag.get('roi_offset', res.get('roi_offset', (0, 0)))
        H, W = diag.get('roi_shape', (0, 0))
        vfield = res.get('vfield_mean')
        if (H == 0 or W == 0) and vfield is not None:
            H, W = vfield.shape[:2]
        speed = (np.sqrt(vfield[..., 0] ** 2 + vfield[..., 1] ** 2) * v_conv
                 if vfield is not None else None)
        _spd_hi = (float(np.percentile(speed[speed > 0], 99))
                   if speed is not None and np.any(speed > 0) else 1.0)

        def _arr(x):
            return np.asarray(x, dtype=float) if x is not None else None
        cl_init = _arr(diag.get('cl_initial'))
        cl_coarse = _arr(diag.get('cl_coarse'))
        cl_ref = _arr(res.get('cl_used'))
        offs_raw = diag.get('offsets_raw')
        offs_sm = diag.get('offsets_smooth')

        # radii (px) + fit velocities (px/frame)
        R_free_px = float(res.get('R_fit_free', np.nan))
        R_kl_px = float(res.get('R_kloosterman', np.nan))
        v0_fix = float(res.get('v0_fixed', np.nan))
        v0_free = float(res.get('v0_free', np.nan))
        R_free_ratio = (R_free_px / r_tile
                        if (np.isfinite(R_free_px) and r_tile > 0) else np.nan)

        # ── quality verdict (suptitle across all pages) ─────────────────
        bc = np.asarray(res.get('bin_centers', []), float)
        pv = np.abs(np.asarray(res.get('profile_v', []), float)) * v_conv
        bc_um = bc * px_um
        fin = np.isfinite(pv)
        in_win = np.abs(bc) <= max(r_tile, 1e-6)
        cov = float((fin & in_win).sum()) / float(max(in_win.sum(), 1))
        # Measure the wall shell at the FITTED radius when the free-R fit found a
        # wider-but-bounded vessel (segmentation under-measured R).  Otherwise an
        # under-segmented parabola reads as "blunt" only because the R_seg shell
        # sits on the vessel CORE, not its wall (e.g. R_seg=13µm on a real 25µm
        # vessel).  Keep R_seg when the free fit is absent or actually railing the
        # gate (ratio ≥ 1.95), where the fitted R is itself untrustworthy.
        R_wall_um = R_prior_um
        if (np.isfinite(R_free_ratio) and 1.15 < R_free_ratio < 1.95
                and np.isfinite(R_free_px) and R_free_px > 0):
            R_wall_um = R_free_px * px_um
        core = fin & (np.abs(bc_um) <= 0.30 * R_wall_um)
        shell = fin & (np.abs(bc_um) >= 0.80 * R_wall_um) & (np.abs(bc_um) <= 1.05 * R_wall_um)
        v_peak = (float(np.nanmax(pv[core])) if core.any()
                  else (float(np.nanmax(pv[fin])) if fin.any() else np.nan))
        v_edge = float(np.nanmean(pv[shell])) if shell.any() else np.nan
        taper = (1.0 - v_edge / v_peak) if (np.isfinite(v_peak) and v_peak > 1e-9
                                            and np.isfinite(v_edge)) else np.nan
        cropped = cov < 0.60
        # RAILING = the free-R fit actually rails the display gate ([0.5,2.0]·R);
        # a moderate 1.3–1.9× is a legitimately wider-than-segmented vessel, not
        # a runaway fit (neighbour bleed is already cut upstream).  (taper above is
        # kept as an informational descriptor only — no BLUNT flag: it false-failed
        # under-segmented but otherwise clean vessels.)
        railing = np.isfinite(R_free_ratio) and R_free_ratio >= 1.95
        n_valid_kl = int(res.get('Q_kloosterman_n_valid', 0) or 0)
        # Split the verdict by WHAT each flag invalidates.  CROP starves every
        # estimator of data → always a hard failure.  RAILING describes the R-FIT
        # geometry: it invalidates Q only when the canonical estimator IS an R-fit
        # (Pois free-/mask-R).  When the reported Q is Kloosterman — per-slice,
        # independent of the free-R blow-up — with enough valid slices, RAILING is
        # an advisory NOTE, not a failure.  (E.g. a clean central peak with low
        # side-shoulders that don't trip neighbour rejection but still drag free-R
        # to >2×: the Kloosterman Q is fine.)
        hard, soft = [], []
        if cropped:
            hard.append('CROP')
        _geom = ['RAILING'] if railing else []
        if best_key == 'kl' and n_valid_kl >= 3:
            soft += _geom
        else:
            hard += _geom
        flags = hard + soft
        if not flags:
            verdict = 'PASS'
        elif hard:
            verdict = ' + '.join(hard) + (
                f"  ·  note: {' + '.join(soft)}" if soft else '')
        else:
            verdict = 'PASS  ·  note: ' + ' + '.join(soft)
        vcol = '#2ca02c' if not hard else '#d62728'

        # ── Q estimators (nL/s) ─────────────────────────────────────────
        Q_kl = float(res.get('Q_kloosterman', np.nan)) * q_conv
        Q_plug = float(res.get('Q_plug_mean', np.nan)) * q_conv
        Q_pois_mask = float(res.get('Q_pois_mean', np.nan)) * q_conv
        Q_pois_free = ((abs(v0_free) * np.pi * R_free_px ** 2 / 2.0) * q_conv
                       if (np.isfinite(v0_free) and np.isfinite(R_free_px) and R_free_px > 0)
                       else np.nan)
        Q_canon = {'kl': Q_kl, 'pois_free': Q_pois_free,
                   'pois_seg': Q_pois_mask}.get(best_key, Q_kl)
        canon_name = {'kl': 'Kloosterman', 'pois_free': 'Pois free-R',
                      'pois_seg': 'Pois mask-R'}.get(best_key, 'Kloosterman')

        fig = plt.figure(figsize=(15, 19), dpi=100)
        fig.suptitle(
            f"OF pipeline — edge ({u}, {v}) · tile {tile_id} · reported Q = "
            f"{abs(Q_canon):.3f} nL/s ({canon_name}) · "
            f"R_seg = {R_prior_um:.1f} µm · {verdict}",
            fontsize=13, color=vcol, fontweight='bold', y=0.992)

        def _draw_speed(ax):
            if speed is not None:
                ax.imshow(np.ma.masked_where(speed <= 0, speed), cmap='hot',
                          vmin=0, vmax=_spd_hi, origin='upper')
            ax.set_aspect('equal')

        def _orient(ax, h, w):
            """Rotate a spatial panel from the raw tile-camera frame into the
            mosaic frame the main viewer shows (the stitcher's per-tile affine
            is a rotation) so the crops match the viewer's orientation.  Applies
            one affine to every artist (image, lines, quiver collections,
            cross-section arrow patches) and re-limits from the rotated corners
            (y inverted = row-down)."""
            try:
                import matplotlib.transforms as _mtx
                _L = self._of_geom.tiles[int(tile_id)].A_full[:2, :2]
                _aff = _mtx.Affine2D.from_values(
                    _L[1, 1], _L[0, 1], _L[1, 0], _L[0, 0], 0.0, 0.0)
                for _a in (list(ax.images) + list(ax.lines)
                           + list(ax.collections) + list(ax.patches)):
                    _a.set_transform(_aff + ax.transData)
                    # Quiver (and other offset collections) position their
                    # glyphs via a SEPARATE offset transform — rotate that too,
                    # or the arrows detach from the rotated image.
                    if hasattr(_a, 'set_offset_transform'):
                        try:
                            _a.set_offset_transform(_aff + ax.transData)
                        except Exception:
                            pass
                _cor = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
                _xt = _L[1, 1] * _cor[:, 0] + _L[1, 0] * _cor[:, 1]
                _yt = _L[0, 1] * _cor[:, 0] + _L[0, 0] * _cor[:, 1]
                _pad = 0.05 * max(np.ptp(_xt), np.ptp(_yt), 1.0)
                ax.set_xlim(_xt.min() - _pad, _xt.max() + _pad)
                ax.set_ylim(_yt.max() + _pad, _yt.min() - _pad)
                ax.set_aspect('equal')
                return True
            except Exception:
                return False

        # One tall SCROLLABLE figure, 3×3, consolidated to the non-redundant
        # panels.  Dropped (redundant / broken): the mean-OF-field quiver panel
        # (broken auto-scaled vectors + duplicate of the speed heatmap), the
        # raw→smoothed pointwise scatter (re-draws the binned profile), the
        # mask+tangent panel, the CL-offset plot, per-slice R, and the second
        # (video-crop) radii panel — one vessel-image panel is enough.
        gs = fig.add_gridspec(4, 3, hspace=0.42, wspace=0.26,
                              height_ratios=[1, 1, 1, 1.3],
                              top=0.962, bottom=0.03, left=0.06, right=0.975)

        # analysed (kept) vs full (untrimmed) centerline arc lengths (µm).
        def _arclen(a):
            a = _arr(a)
            return (float(np.sum(np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))))
                    if (a is not None and len(a) >= 2) else np.nan)
        cl_kept = _arr(data.get('centerline_rc'))
        cl_all = _arr(data.get('centerline_full_rc'))
        _kept_um = _arclen(cl_kept) * px_um
        _full_um = _arclen(cl_all) * px_um
        _pct = (100.0 * _kept_um / _full_um
                if (np.isfinite(_kept_um) and np.isfinite(_full_um)
                    and _full_um > 0) else np.nan)

        # (0,0) ROI LOCATOR — zoomed to the vessel, trim shown
        ax = fig.add_subplot(gs[0, 0])
        frames = data.get('frames'); Hf = Wf = None
        if frames is not None:
            fr = np.asarray(frames)
            if fr.ndim == 3 and fr.shape[0]:
                mf = fr.mean(axis=0); Hf, Wf = mf.shape[:2]
                loi, hii = np.percentile(mf, (1, 99))
                ax.imshow(mf, cmap='gray', vmin=loi, vmax=hii, origin='upper')
                ax.add_patch(Rectangle((c0, r0), W, H, fill=False,
                                       edgecolor='lime', lw=2.0))
                if cl_all is not None and len(cl_all) >= 2:
                    ax.plot(cl_all[:, 1], cl_all[:, 0], color='white', lw=2.4,
                            ls=(0, (4, 2)), alpha=0.9, label='full vessel CL')
                if cl_kept is not None and len(cl_kept) >= 2:
                    ax.plot(cl_kept[:, 1], cl_kept[:, 0], color='cyan', lw=1.6,
                            label='analysed CL')
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=6.5, loc='upper right', facecolor='black',
                              labelcolor='white', framealpha=0.5)
        if Hf:
            _orient(ax, Hf, Wf)
            try:
                _Lz = self._of_geom.tiles[int(tile_id)].A_full[:2, :2]
                _bc = np.array([[c0, r0], [c0 + W, r0],
                                [c0, r0 + H], [c0 + W, r0 + H]], float)
                _bx = _Lz[1, 1] * _bc[:, 0] + _Lz[1, 0] * _bc[:, 1]
                _by = _Lz[0, 1] * _bc[:, 0] + _Lz[0, 0] * _bc[:, 1]
                _pz = 0.35 * max(np.ptp(_bx), np.ptp(_by), 1.0)
                ax.set_xlim(_bx.min() - _pz, _bx.max() + _pz)
                ax.set_ylim(_by.max() + _pz, _by.min() - _pz)
            except Exception as _e:
                _note_suppressed("OF pipeline figure: tile-frame zoom", _e)
        else:
            ax.text(0.5, 0.5, 'no frames', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
        _crop_str = (f"CL kept {_kept_um:.0f}/{_full_um:.0f} µm ({_pct:.0f}%)"
                     if np.isfinite(_pct) else f"CL {_kept_um:.0f} µm")
        ax.set_title(f"ROI locator — green box = CROP (mosaic orient.)\n"
                     f"{_crop_str} · ROI {W}×{H} px", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        # (0,1:) radial profile + 3 fits + gate — widened across the two right
        # cells (the old (0,1) "vessel flow + cyan fan" panel was redundant with
        # the 3D per-slice-parabola panel below, which shows the fitted profile on
        # the ACTUAL curved vessel; dropped it and gave the space to the profile).
        ax = fig.add_subplot(gs[0, 1:])
        _neigh = np.asarray(diag.get('neighbour_mask', []), dtype=bool)
        _cent = np.asarray(diag.get('central_mask', []), dtype=bool)
        if _neigh.shape == bc_um.shape and _neigh.any():
            # Split measured points: central vessel (kept for free-R fit) vs
            # bleed-in neighbour (rejected past the first velocity valley).
            _kept = fin & ~_neigh
            _rej = fin & _neigh
            ax.plot(bc_um[_kept], pv[_kept], 'o', ms=4, color='#1f77b4',
                    label='measured (central)')
            ax.plot(bc_um[_rej], pv[_rej], 'x', ms=5, color='#999999', mew=1.4,
                    label='rejected (neighbour bleed)')
            if _cent.shape == bc_um.shape and _cent.any():
                _ci = np.where(_cent)[0]
                _lo_um = float(bc_um[_ci.min()]); _hi_um = float(bc_um[_ci.max()])
                _xmin = float(bc_um.min()) - 1.0; _xmax = float(bc_um.max()) + 1.0
                ax.axvspan(_xmin, _lo_um, color='gray', alpha=0.12)
                ax.axvspan(_hi_um, _xmax, color='gray', alpha=0.12,
                           label='rejected region')
        else:
            ax.plot(bc_um[fin], pv[fin], 'o', ms=4, color='#1f77b4',
                    label='measured')
        rr = np.linspace(-R_prior_um, R_prior_um, 100)
        if np.isfinite(v0_fix):
            ax.plot(rr, np.clip(abs(v0_fix) * v_conv * (1 - (rr / R_prior_um) ** 2), 0, None),
                    '-', color='#2ca02c', lw=1.8, label=f'mask-R fit ({R_prior_um:.0f} µm)')
        if np.isfinite(R_free_px) and R_free_px > 0 and np.isfinite(v0_free):
            R_free_um = R_free_px * px_um
            rr2 = np.linspace(-R_free_um, R_free_um, 100)
            ax.plot(rr2, np.clip(abs(v0_free) * v_conv * (1 - (rr2 / R_free_um) ** 2), 0, None),
                    '--', color='#d62728', lw=1.8, label=f'free-R fit ({R_free_um:.0f} µm)')
        lo = float(diag.get('kl_r_slice_lo', 0.5)); hi = float(diag.get('kl_r_slice_hi', 2.0))
        ax.axvspan(lo * R_prior_um, hi * R_prior_um, color='orange', alpha=0.06)
        ax.axvspan(-hi * R_prior_um, -lo * R_prior_um, color='orange', alpha=0.06,
                   label=f'fitted-R gate [{lo:.1f},{hi:.1f}]·R')
        for xv in (-R_prior_um, R_prior_um):
            ax.axvline(xv, color='magenta', ls=':', lw=1.0)
        _pk = [np.nanpercentile(pv[fin], 97)] if fin.any() else []
        if np.isfinite(v0_fix):
            _pk.append(abs(v0_fix) * v_conv)
        if np.isfinite(v0_free):
            _pk.append(abs(v0_free) * v_conv)
        if _pk:
            ax.set_ylim(0, 1.3 * max(_pk))
        ax.set_xlabel('signed ρ (µm)'); ax.set_ylabel('|v| (µm/s)')
        ax.set_title(f"radial profile + fits — taper {taper:.2f}, "
                     f"R_free/R_seg {R_free_ratio:.2f}", fontsize=9)
        ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=0.3)

        # ── Kloosterman per-slice prep ──
        kl = diag.get('kl') or {}
        psQ = np.asarray(kl.get('per_slice_Q', []), float)
        psR = np.asarray(kl.get('per_slice_R', []), float)
        vmask = np.asarray(kl.get('valid_mask', []), bool)
        reasons = list(kl.get('per_slice_reason', []))
        Nkl = len(psQ)
        good = vmask if vmask.size == Nkl else np.isfinite(psQ)
        xi = np.arange(Nkl)
        psQ_nl = psQ * q_conv; psR_um = psR * px_um
        Q_char = float(res.get('Q_kloosterman', np.nan)) * q_conv
        V_char = float(kl.get('V_CL_char', np.nan)) * v_conv
        R_char_um = float(kl.get('R_char', np.nan)) * px_um
        Qmed = float(np.nanmedian(psQ_nl[np.isfinite(psQ_nl)])) if np.isfinite(psQ_nl).any() else np.nan

        # (1,0) CL refinement — graph CL vs OF ridge-tracked CL
        ax = fig.add_subplot(gs[1, 0]); _draw_speed(ax)
        if cl_init is not None:
            ax.plot(cl_init[:, 1], cl_init[:, 0], color='cyan', lw=1.4, ls=':',
                    label='graph CL (initial)')
        if cl_ref is not None:
            ax.plot(cl_ref[:, 1], cl_ref[:, 0], color='lime', lw=1.8,
                    label='ridge-tracked CL')
        ax.legend(fontsize=7, loc='upper right', facecolor='black',
                  labelcolor='white', framealpha=0.5)
        _sh = res.get('cl_shift')
        ax.set_title("centerline refinement — OF ridge-tracking (mosaic orient.)\n"
                     + (f"mean shift {_sh:+.1f} px onto the flow ridge"
                        if _sh is not None else ""), fontsize=9)
        _orient(ax, H, W); ax.set_xticks([]); ax.set_yticks([])

        # (1,1) per-slice Q + median-band gate  (the spatial per-slice accept/
        # reject + fitted half-widths now live in the 3D panel below — this is
        # the quantitative gate view, no longer duplicated as a 2D half-widths
        # panel).
        ax = fig.add_subplot(gs[1, 1])
        # Q is a flow-rate magnitude → orient the whole panel positive (same sign
        # convention as the Q(t) panel: flip the trace by the median sign so the
        # points, median line, band and Q_char all read positive together).
        _sgn = -1.0 if (np.isfinite(Qmed) and Qmed < 0) else 1.0
        psQ_p = psQ_nl * _sgn
        Qmed_p = Qmed * _sgn
        Q_char_p = Q_char * _sgn
        if Nkl:
            ax.plot(xi[good], psQ_p[good], 'o', ms=5, color='#2ca02c', label='accepted')
            bad = (~good) & np.isfinite(psQ_p)
            ax.plot(xi[bad], psQ_p[bad], 'x', ms=6, color='#999', label='rejected')
            for bi in list(xi[bad])[:5]:
                if bi < len(reasons):
                    ax.annotate(str(reasons[bi]).split('(')[0], (bi, psQ_p[bi]),
                                fontsize=6, color='#666', ha='center')
        if np.isfinite(Qmed_p):
            ax.axhline(Qmed_p, color='k', lw=1.0, label=f'median {Qmed_p:.3f}')
            ax.axhspan(0.5 * Qmed_p, 1.5 * Qmed_p, color='green', alpha=0.07,
                       label='band [0.5,1.5]·med')
        if np.isfinite(Q_char_p):
            ax.axhline(Q_char_p, color='#1f77b4', ls='--', lw=1.6, label=f'Q_char {Q_char_p:.3f}')
        ax.set_xlabel('slice'); ax.set_ylabel('Q_slice (nL/s)')
        ax.set_title("per-slice Q + median-band gate", fontsize=9)
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

        # (1,2) Q estimator bar
        ax = fig.add_subplot(gs[1, 2])
        bars = [('plug', Q_plug, '#7f7f7f'), ('Pois mask-R', Q_pois_mask, '#2ca02c'),
                ('Pois free-R', Q_pois_free, '#d62728'), ('Kloosterman', Q_kl, '#1f77b4')]
        names = [b[0] for b in bars]
        vals = [abs(b[1]) if np.isfinite(b[1]) else 0.0 for b in bars]
        cols = [b[2] for b in bars]
        edgs = ['black' if n == canon_name else 'none' for n in names]
        lws = [2.5 if n == canon_name else 0 for n in names]
        ax.bar(range(len(names)), vals, color=cols, edgecolor=edgs, linewidth=lws)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('|Q| (nL/s)')
        ax.set_title(f"Q estimators — canonical = {canon_name}", fontsize=9)
        ax.grid(alpha=0.25, axis='y')

        # (2,0) FULL Q(t) time series + harmonic fit  (not the cycle-average).
        # CONSISTENT with the reported canonical Q: the per-frame Q_pois_t is the
        # mask-R Poiseuille SHAPE (fit v0(t) with R fixed to R_seg); its absolute
        # scale (R_seg) generally differs from the canonical estimator's radius
        # (Kloosterman uses per-slice fitted R_char).  Those are INDEPENDENT paths
        # — Q_pois_t never touches Kloosterman — so raw they disagree by ~(R/ R_seg)².
        # We therefore rescale the waveform by pois_seg_factor (= Q_canon / mean
        # Q_pois) so its mean lands on the reported line: the temporal modulation is
        # the real per-frame OF, the magnitude is the canonical value.  (The raw
        # per-estimator magnitudes are compared side-by-side in the estimator bar.)
        ax = fig.add_subplot(gs[2, 0])
        _psf = float(data.get('pois_seg_factor', 1.0) or 1.0)
        Qraw = np.asarray(res.get('Q_pois_t', []), float) * q_conv * _psf
        if Qraw.size and np.isfinite(np.nanmean(Qraw)) and np.nanmean(Qraw) < 0:
            Qraw = -Qraw                    # orient positive (Q is a magnitude)
        t_s = np.arange(Qraw.size) * frame_dt_s
        mean_q = float(np.nanmean(Qraw)) if Qraw.size else np.nan
        a0 = amp1 = np.nan
        fitsig = np.array([])
        if Qraw.size > 8 and np.isfinite(f0_hz) and f0_hz > 0:
            try:
                from ..analyze.harmonic import fit_harmonics
                hr = fit_harmonics(Qraw, frame_dt=frame_dt_s, f0=f0_hz,
                                   K=3, include_dc=True)
                a0 = float(hr.get('a0', np.nan))
                amp1 = float(hr.get('amp1', np.nan))
                fitsig = np.asarray(hr.get('signal', []), float)
            except Exception as _e:
                _note_suppressed("OF pipeline figure: harmonic fit of Q(t)", _e)
            ax.plot(t_s, Qraw, color='#c2c2c2', lw=0.7,
                    label='Q(t) (canonical scale)')
            if fitsig.size == Qraw.size:
                ax.plot(t_s, fitsig, color='#1f77b4', lw=1.6,
                        label='harmonic fit (DC+3H)')
            if np.isfinite(Q_canon):
                # mean now equals the reported value by construction
                ax.axhline(abs(Q_canon), color='#d62728', ls='-.', lw=1.3,
                           label=f'reported = mean {abs(Q_canon):.3f} ({canon_name})')
            ax.legend(fontsize=6.5, loc='upper right')
        else:
            ax.text(0.5, 0.5, 'no Q(t)', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
        ax.set_xlabel('time (s)'); ax.set_ylabel('Q (nL/s)')
        ax.set_title(f"Q(t) + harmonic fit (canonical scale) — mean {mean_q:.3f}, "
                     f"H1 {amp1:.3f} nL/s · f₀={f0_hz:.2f} Hz", fontsize=8.5)
        ax.grid(alpha=0.3)

        # (2,1:) verdict + compact Kloosterman derivation (spans two cells)
        ax = fig.add_subplot(gs[2, 1:]); ax.axis('off')

        def _f(x, fmt='{:.2f}'):
            return fmt.format(x) if (x is not None and np.isfinite(x)) else '—'
        cats = {}
        for i, rs in enumerate(reasons):
            if i < len(good) and good[i]:
                cats['accepted'] = cats.get('accepted', 0) + 1
            else:
                key = str(rs).split('(')[0]
                cats[key] = cats.get(key, 0) + 1
        cat_str = ", ".join(f"{k} {n}" for k, n in
                            sorted(cats.items(), key=lambda kv: -kv[1])[:4])
        # When Kloosterman was NOT used, say so loudly + why it failed.
        _kl_failed = (canon_name != 'Kloosterman')
        _kl_reason = str(kl.get('reason', '?'))
        _fail_line = (f"  ⚠ KLOOSTERMAN FAILED ({int(good.sum())}/{Nkl} slices, "
                      f"{_kl_reason}) → fell back to {canon_name}\n"
                      f"    why: {cat_str}\n"
                      if _kl_failed else "")
        txt = (
            f"VERDICT:  {verdict}\n"
            + _fail_line +
            f"  coverage {cov*100:.0f}%   taper {_f(taper)}   "
            f"R_free/R_seg {_f(R_free_ratio)}\n"
            f"  Kloosterman: {int(good.sum())}/{Nkl} slices kept\n"
            f"  reported Q = {abs(Q_canon):.4f} nL/s ({canon_name})\n"
            f"  R_seg {R_prior_um:.1f} µm    R_inferred {_f(R_free_px*px_um,'{:.1f}')} µm\n"
            + ("  → mask-R and fitted-R agree — both reliable\n"
               if not (set(flags) & {'CROP', 'RAILING'})
               else "  → trust mask-R; fitted-R unreliable here\n")
            + "\nKloosterman derivation\n"
            "──────────────────────\n"
            f"  incl: |v| > {diag.get('kl_v_incl_frac', 0.25):.2f}·v_peak"
            f" (contiguous extent → wall)\n"
            f"  R gate [{lo:.1f},{hi:.1f}]·R_seg (loose; median-Q rejects)\n"
            f"  outlier: |Q| in [0.5,1.5]·median, sign-matched\n"
            f"  outcomes: {cat_str}\n"
            f"  Q_char {_f(Q_char,'{:.4f}')}   V_CL {_f(V_char,'{:.0f}')}"
            f"   R_char {_f(R_char_um,'{:.1f}')} µm"
        )
        ax.text(0.0, 1.0, txt, transform=ax.transAxes, ha='left', va='top',
                family='monospace', fontsize=8,
                bbox=dict(boxstyle='round',
                          facecolor=('#e8f5e9' if not hard else '#fdecea'),
                          edgecolor=vcol, lw=1.5))

        # (3,:) 3D per-slice parabolic fits on the ACTUAL curved vessel
        # (Kloosterman-thesis Fig 4.11): each cross-section's fitted parabola is
        # drawn at its real spatial position along the perpendicular, so the
        # vessel's real curvature shows in the x–y (col,row) plane.  The vessel
        # speed image sits on the z=0 floor, the cyan centerline curves through
        # it, and the white ±R_fit walls trace the full vessel extent implied by
        # the parabola radii.  Accepted = coloured, rejected = gray.
        ax = fig.add_subplot(gs[3, :], projection='3d')
        psV = np.asarray(kl.get('per_slice_v_CL', []), float)
        profs = np.asarray(kl.get('per_slice_profiles', []), float)
        rho_g = np.asarray(kl.get('rho', []), float)
        drew3d = False
        if (cl_ref is not None and centerline_geometry is not None
                and profs.ndim == 2 and profs.shape[0] == Nkl and rho_g.size
                and len(cl_ref) >= 2 and Nkl >= 2 and vfield is not None):
            import matplotlib.cm as _cm
            clr = np.asarray(cl_ref, float)           # (N, 2) = (row, col)
            _s3, _tg3, nm3 = centerline_geometry(clr)  # normals in (row, col)
            # z-scale from the measured cross-section velocities
            _pf = np.abs(profs[np.isfinite(profs)]) * v_conv
            _zmax = float(np.nanpercentile(_pf, 99)) if _pf.size else 1.0
            # floor: grayscale ROI intensity image — extends BEYOND the vessel
            # into the tissue, so the inferred radius (parabolas + ±R_fit walls)
            # can be compared against the vessel's actual edges in the image.
            try:
                import matplotlib.pyplot as _plt
                _fm = diag.get('frame_mean')
                _fm = (np.asarray(_fm, float) if _fm is not None
                       else np.sqrt(vfield[..., 0] ** 2 + vfield[..., 1] ** 2))
                _st = max(1, min(_fm.shape[:2]) // 80)   # ≤~80px for a fast surface
                _fs = _fm[::_st, ::_st]
                _lo, _hi = np.percentile(_fs, (1, 99))
                _g = np.clip((_fs - _lo) / max(_hi - _lo, 1e-6), 0, 1)
                _Yg, _Xg = np.mgrid[0:_fs.shape[0], 0:_fs.shape[1]]
                ax.plot_surface(_Xg * _st * px_um, _Yg * _st * px_um,
                                np.zeros_like(_g), facecolors=_plt.cm.gray(_g),
                                rstride=1, cstride=1, shade=False, linewidth=0,
                                antialiased=False, alpha=0.4, zorder=0)
            except Exception:
                pass
            n_acc = max(int(good.sum()), 1); _ai = 0
            wp, wm = [], []
            for i in range(Nkl):
                acc = bool(i < len(good) and good[i])
                col = (_cm.turbo(_ai / n_acc) if acc else '#b0b0b0')
                if acc:
                    _ai += 1
                pi = clr[min(i, len(clr) - 1)]; ni = nm3[min(i, len(nm3) - 1)]
                # measured cross-section points at their real (col,row)
                pr = profs[i]; fm = np.isfinite(pr)
                if fm.any():
                    pts = pi[None, :] + rho_g[fm][:, None] * ni[None, :]
                    ax.scatter(pts[:, 1] * px_um, pts[:, 0] * px_um,
                               np.abs(pr[fm]) * v_conv, color=col,
                               s=(9 if acc else 5), alpha=(0.95 if acc else 0.3),
                               edgecolors='none', zorder=(5 if acc else 1))
                # fitted parabola along the real perpendicular
                if (acc and i < len(psV) and i < len(psR)
                        and np.isfinite(psV[i]) and np.isfinite(psR[i]) and psR[i] > 0):
                    rr = np.linspace(-psR[i], psR[i], 40)
                    pos = pi[None, :] + rr[:, None] * ni[None, :]
                    zf = np.clip(abs(psV[i]) * (1 - (rr / psR[i]) ** 2), 0, None) * v_conv
                    ax.plot(pos[:, 1] * px_um, pos[:, 0] * px_um, zf,
                            color=col, lw=1.6, alpha=0.95, zorder=5)
                    wp.append(pi + psR[i] * ni); wm.append(pi - psR[i] * ni)
            # centerline + ±R_fit extent walls, lifted just above the floor
            # heatmap so they aren't buried under it.
            _zoff = 0.04 * max(_zmax, 1.0)
            ax.plot(clr[:, 1] * px_um, clr[:, 0] * px_um, np.full(len(clr), _zoff),
                    color='lime', lw=2.6, alpha=1.0, zorder=6, label='centerline')
            if len(wp) >= 2:
                wp = np.asarray(wp); wm = np.asarray(wm)
                ax.plot(wp[:, 1] * px_um, wp[:, 0] * px_um, np.full(len(wp), _zoff),
                        color='magenta', lw=2.2, alpha=1.0, zorder=6)
                ax.plot(wm[:, 1] * px_um, wm[:, 0] * px_um, np.full(len(wm), _zoff),
                        color='magenta', lw=2.2, alpha=1.0, zorder=6,
                        label='vessel extent (±R_fit)')
            ax.legend(fontsize=7, loc='upper left')
            ax.set_xlabel('col (µm)'); ax.set_ylabel('row (µm)')
            ax.set_zlabel('|v| (µm/s)')
            ax.set_zlim(0, 1.05 * max(_zmax, 1.0))
            ax.view_init(elev=26, azim=-72)
            drew3d = True
        if not drew3d:
            ax.text2D(0.5, 0.5, 'no per-slice profile data',
                      transform=ax.transAxes, ha='center', va='center', color='gray')
        ax.set_title(f"per-slice parabolic fits on the real curved vessel — "
                     f"coloured = accepted ({int(good.sum())}), gray = rejected · "
                     f"magenta = ±R_fit extent  (reject if R² < "
                     f"{diag.get('kl_min_r2', 0.5):.2f})", fontsize=9)

        # ── save PNG + show in a scrollable window (with the mpl toolbar) ──
        try:
            from pathlib import Path as _Path
            _out = _Path.home() / 'Desktop' / 'of_diagnostics'
            _out.mkdir(parents=True, exist_ok=True)
            fig.savefig(_out / f'OF_pipeline_tile{tile_id}_edge{u}-{v}.png',
                        dpi=130, facecolor='white')
        except Exception as _e:
            print(f"  (pipeline PNG save skipped: {_e})")
        self._show_scrollable_figure(fig, position='top-right')



    def _show_scrollable_figure(self, fig, position=None):
        """Show one tall Figure in a scrollable Qt dialog with the navigation
        toolbar (zoom / pan / home / save).  The canvas keeps its full rendered
        size and a QScrollArea provides vertical scrolling, so the stacked
        pipeline sections read top→bottom like a document.  Falls back to
        show_figure if the Qt path is unavailable."""
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg, NavigationToolbar2QT)
            from qtpy.QtWidgets import (QDialog, QVBoxLayout, QScrollArea,
                                        QApplication, QSizePolicy)
            from qtpy.QtCore import Qt, QSize
            canvas = FigureCanvasQTAgg(fig)
            w = int(fig.get_figwidth() * fig.dpi)
            h = int(fig.get_figheight() * fig.dpi)
            # setFixedSize (not setMinimumSize) — matplotlib's canvas
            # overrides sizeHint() after draw() and would shrink the
            # canvas back to the dialog viewport, which collapses the
            # scroll area and makes the bars disappear a moment after
            # the dialog opens.  Fixed size prevents that race.
            canvas.setFixedSize(w, h)
            dialog = QDialog()
            dialog.setWindowTitle(
                fig._suptitle.get_text().split(chr(10))[0]
                if fig._suptitle else 'OF pipeline')
            dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMinMaxButtonsHint)
            lay = QVBoxLayout(dialog); lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(NavigationToolbar2QT(canvas, dialog))
            scroll = QScrollArea()
            scroll.setWidgetResizable(False)
            # Force scrollbars always visible so it's obviously scrollable.
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            # Kill the scroll-area's own sizeHint so it doesn't demand the
            # canvas's full size and drag the dialog to fig_h — which is
            # what macOS was doing (dialog grew to figure, no scroll needed
            # → bars flashed then vanished).  Small size hint = the outer
            # dialog.resize() actually wins.
            scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            scroll.setMinimumSize(QSize(400, 300))
            scroll.setWidget(canvas); lay.addWidget(scroll)
            screen = QApplication.primaryScreen()
            avail = screen.availableGeometry() if screen is not None else None
            if avail is not None:
                vw = min(w + 26, int(avail.width() * 0.62))
                vh = min(h + 60, int(avail.height() * 0.85))
            else:
                vw, vh = min(w + 26, 1200), 900
            # setFixedSize (not resize) — locks the dialog against Qt's
            # sizeHint-based auto-grow so it can't expand to the canvas.
            dialog.setFixedSize(vw, vh)
            canvas.draw(); dialog.show()
            # After show, restore resizability so the user can drag corners
            # if they want a bigger view.
            dialog.setMinimumSize(QSize(400, 300))
            dialog.setMaximumSize(QSize(16777215, 16777215))
            if position == 'top-right' and avail is not None:
                try:
                    QApplication.processEvents()
                    fw = dialog.frameGeometry().width()
                    dialog.move(int(avail.x() + avail.width() - fw - 12),
                                int(avail.y() + 12))
                except Exception:
                    pass
            _d = [d for d in getattr(self, '_pipeline_dialogs', []) if d.isVisible()]
            _d.append(dialog); self._pipeline_dialogs = _d
        except Exception as _e:
            from .mosaic import show_figure
            print(f"  (scroll dialog failed, using plain canvas: {_e})")
            show_figure(fig, position=position)


    def _show_fig_with_toolbar(self, fig, position=None):
        """Show a matplotlib Figure in a Qt dialog that includes the standard
        navigation toolbar (zoom / pan / home / save) — which the bare
        ``show_figure`` embed lacks.  Keeps the dialog alive on the viewer and
        falls back to ``show_figure`` if the Qt toolbar path is unavailable."""
        try:
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg, NavigationToolbar2QT)
            from qtpy.QtWidgets import QDialog, QVBoxLayout, QApplication
            from qtpy.QtCore import Qt
            canvas = FigureCanvasQTAgg(fig)
            dialog = QDialog()
            _t = (fig._suptitle.get_text().split(chr(10))[0]
                  if fig._suptitle else 'Figure')
            dialog.setWindowTitle(_t)
            dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMinMaxButtonsHint)
            lay = QVBoxLayout(dialog); lay.setContentsMargins(0, 0, 0, 0)
            toolbar = NavigationToolbar2QT(canvas, dialog)
            lay.addWidget(toolbar); lay.addWidget(canvas)
            dialog.resize(int(fig.get_figwidth() * fig.dpi),
                          int(fig.get_figheight() * fig.dpi) + 40)
            canvas.draw(); dialog.show()
            if position == 'top-right':
                try:
                    QApplication.processEvents()
                    screen = QApplication.primaryScreen()
                    if screen is not None:
                        avail = screen.availableGeometry()
                        fw = dialog.frameGeometry().width()
                        n = len(getattr(self, '_pipeline_dialogs', []))
                        dialog.move(int(avail.x() + avail.width() - fw - 12 - 26 * n),
                                    int(avail.y() + 12 + 26 * n))
                except Exception:
                    pass
            _d = [d for d in getattr(self, '_pipeline_dialogs', []) if d.isVisible()]
            _d.append(dialog)
            self._pipeline_dialogs = _d
        except Exception as _e:
            from .mosaic import show_figure
            print(f"  (toolbar dialog failed, using plain canvas: {_e})")
            show_figure(fig, position=position)


    def _show_of_pipeline_figures(self):
        """Re-open the full OF pipeline walkthrough for the last fresh OF run."""
        data = getattr(self, '_last_of_pipeline', None)
        if not data:
            self._info_box.setPlainText(
                "Run 'Compute fresh PIV/OF' on an edge first.")
            return
        try:
            self._render_of_pipeline(data)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._info_box.setPlainText(
                f"OF pipeline figure failed: {type(e).__name__}: {e}")


    def _degree2_chain_edges(self, u, v):
        """Maximal degree-2 chain of edges containing (u, v), ordered and
        connected.  A vessel is subdivided into segments by trace nodes; the
        interior nodes of a single vessel are degree-2 (pass-through, where KCL
        is trivially satisfied).  Merging the chain lets fresh OF run on the
        whole vessel instead of one short segment — more cross-sections, higher
        SNR, one consistent flow direction.  We walk outward from u and from v
        through degree-2 nodes, stopping at the first junction (deg>=3) or leaf.

        Returns an ordered edge list [(n0,n1),(n1,n2),...] with (u,v) inside.
        The tile-FOV trim in the handler clips whatever runs off-screen, so a
        chain that crosses a tile boundary is naturally limited to the visible
        run — no explicit tile logic needed here."""
        UG = self.G.to_undirected() if self.G.is_directed() else self.G

        def walk(prev, cur):
            # extend from `cur`, arriving from `prev`, through degree-2 nodes
            out = []
            while UG.degree(cur) == 2:
                nxt = next((n for n in UG.neighbors(cur) if n != prev), None)
                if nxt is None or nxt == prev:
                    break
                out.append((cur, nxt))
                if nxt in (u, v):            # closed loop guard
                    break
                prev, cur = cur, nxt
            return out

        left = walk(v, u)                    # [(u,x1),(x1,x2),...] away from v
        right = walk(u, v)                   # [(v,y1),(y1,y2),...] away from u
        left_ordered = [(b, a) for (a, b) in reversed(left)]   # ...->x1->u
        return left_ordered + [(u, v)] + right


    def _on_run_optical_flow_clicked(self):
        """Run the unified Farneback OF (`compute_of_profile_and_Q`) on
        the currently-selected edge and pop the editing viewer's full
        6-panel diagnostic — exactly the same figure as Analyze Mode in
        `mosaic_app`, with no graph writes and no napari arrow overlay.

        Strategy: build the `of_result` dict in the shape
        `_analyze_of_display` expects, alias `self.mosaic_graph` to
        `self.G` so the editing-viewer method can read AV refs without
        modification, then call the method as an unbound function.

        This deliberately replaces the earlier `compute_of_qt_local`
        path — that one only returned `Q_t` + `v_mean`, missing the
        velocity field, vessel mask, radial profile, and ROI bounds
        the figure needs."""
        if not self._require_workbench('fresh optical flow'):
            return
        from qtpy.QtWidgets import QApplication
        import time as _time
        if self._selected_edge is None:
            self._info_box.setPlainText(
                "Click an edge first, then press 'Run optical flow'.")
            return
        u, v = self._selected_edge
        # Focus-tile rule.  Priority order:
        #   1. View-tab tile filter (explicit user choice).
        #   2. Currently-loaded video tile (the video that's actually
        #      on screen — fresh OF should match what the user can see).
        #   3. Click-derived "best-quality" tile from the selected edge
        #      (fallback for edges that have stored measurements but no
        #      currently-loaded video).
        # Old code skipped #2 and jumped straight to #3, which made
        # fresh OF run on a different tile's video than the loaded one.
        tile_id = (self.current_tile_filter
                    if self.current_tile_filter is not None
                    else (self._video_tile_id
                           if self._video_tile_id is not None
                           else self._selected_tile_id))
        if tile_id is None:
            self._info_box.setPlainText(
                f"Edge ({u}, {v}) has no PIV measurement on file, so "
                f"there's no recorded tile to use.\n"
                f"Load a tile video first ('Browse tiles' or the "
                f"per-edge button), or set the Tile filter in the View "
                f"tab to choose a tile explicitly.")
            return
        if not self.tiles or self.of_video_dir is None:
            self._info_box.setPlainText(
                "Optical flow needs --tile-positions and --of-video-dir "
                "(or --video-dir as a fallback) "
                "on the CLI (or the corresponding entries in the "
                "config bundle).")
            return
        video_path = self._get_of_video_path_for(int(tile_id))
        if video_path is None:
            self._info_box.setPlainText(
                f"No video found for tile {tile_id} under "
                f"{self.of_video_dir}.")
            return

        # Lazy imports so launch stays light.  Rewired to plexusflow's
        # own coord helpers (graph_to_tile_batch + graph_radius_to_tile_px);
        # pertile's mosaic_to_tile_coords / get_chain_coords aren't shipped
        # in the standalone plexusflow port.
        try:
            from ..analyze.profile import compute_of_profile_and_Q
            from ..analyze.constants import PX_SIZE_UM, FRAME_DT_S
            from ..io.tiff import load_tiff_stack, cut_before_fade
            from ..analyze.mosaic_piv import (
                graph_to_tile_batch, graph_radius_to_tile_px,
                load_tile_geometry, tile_video_shape)
            from ..config import StageConfig
        except Exception as e:
            self._info_box.setPlainText(
                f"Optical-flow imports failed: {type(e).__name__}: {e}")
            return

        # Build a small chain-coord helper — concatenate edge paths for a
        # chain of (u, v) hops, dropping duplicate junctions.  Returns
        # (coords_xy, mean_radius, was_reversed_first) so downstream tile-
        # coord math sees a single ordered centerline.
        def _get_chain_coords_local(G, chain, margin_px=None):
            if not chain:
                return None, float('nan'), False
            all_pts = []
            radii = []
            for i, (uu, vv) in enumerate(chain):
                ed = G.edges[uu, vv]
                pts = ed.get('pts', ed.get('path'))
                if pts is None:
                    continue
                p = np.asarray(pts, dtype=float)
                # Order pts to run u → v (path may be stored either way).
                x0, y0 = float(G.nodes[uu]['x']), float(G.nodes[uu]['y'])
                x1, y1 = float(G.nodes[vv]['x']), float(G.nodes[vv]['y'])
                if (np.hypot(p[0, 0] - x1, p[0, 1] - y1)
                        < np.hypot(p[0, 0] - x0, p[0, 1] - y0)):
                    p = p[::-1]
                if all_pts and len(all_pts[-1]):
                    if np.allclose(all_pts[-1][-1], p[0]):
                        p = p[1:]
                all_pts.append(p)
                # Junction-edge radii excluded (matches pertile convention)
                if len(chain) > 2 and (i == 0 or i == len(chain) - 1):
                    continue
                r = ed.get('radius')
                if r is not None:
                    radii.append(float(r))
            if not all_pts:
                return None, float('nan'), False
            coords_xy = np.concatenate(all_pts, axis=0)
            mean_r = float(np.median(radii)) if radii else \
                     float(np.median([G.edges[uu, vv].get('radius', 5.0)
                                       for uu, vv in chain]))
            # Minimal junction trim.  The old fixed 8px discarded a lot of good
            # vessel length; instead trim just enough to step off the exact
            # junction node (where the CL direction is undefined / kinked) and
            # leave ALL residual end-effect slices — junction bulges, merges,
            # abrupt direction changes — to be rejected downstream by the
            # Kloosterman median-flow-rate band and per-slice validity gates.
            # Flat ~3px (NOT radius-scaled — thick vessels don't need a bigger
            # trim), but fractionally capped so a SHORT vessel (incl. short LARGE
            # ones) is never gutted: never remove more than 15% of the arc per
            # end, and the ≥5-point guard keeps very short vessels whole.
            # ``margin_px=None`` (default) requests this; a number overrides it.
            #
            # Resample to ~1px UNIFORM spacing BEFORE trimming.  The raw graph
            # vessel path can be wildly nonuniform — e.g. two long terminal
            # segments plus a cluster of mid points.  Trimming by dropping whole
            # end POINTS then removes far more than the intended margin: one real
            # vessel had end segments of 9.8 and 15.6px, so a 3px margin dropped
            # the two endpoints and cut 65% of the arc (62→21µm).  Uniform spacing
            # makes the margin remove exactly ~m px per end.
            if len(coords_xy) >= 2:
                _seg = np.diff(coords_xy, axis=0)
                _s = np.concatenate(
                    [[0.0], np.cumsum(np.hypot(_seg[:, 0], _seg[:, 1]))])
                _tot = float(_s[-1])
                if _tot >= 2.0:
                    _n = int(np.clip(round(_tot) + 1,
                                     max(5, len(coords_xy)), 400))
                    _sn = np.linspace(0.0, _tot, _n)
                    coords_xy = np.column_stack([
                        np.interp(_sn, _s, coords_xy[:, 0]),
                        np.interp(_sn, _s, coords_xy[:, 1])])
            if len(coords_xy) > 4:
                seg = np.diff(coords_xy, axis=0)
                arc = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
                total = arc[-1]
                m = margin_px if margin_px is not None else min(3.0, 0.15 * total)
                if m > 0:
                    keep = (arc >= m) & (arc <= total - m)
                    if keep.sum() >= 5:
                        coords_xy = coords_xy[keep]
            return coords_xy, mean_r, False

        self._run_of_btn.setEnabled(False)
        self._info_box.setPlainText(
            f"Running optical flow on edge ({u}, {v}), tile {tile_id} "
            f"…  (~5–10s)")
        QApplication.processEvents()

        try:
            t0 = _time.time()
            # 1. Centerline + radius from the graph (mosaic Cartesian).
            #    Optionally merge the whole degree-2 vessel chain so OF runs on
            #    the full vessel, not just the clicked segment.
            if getattr(self, '_of_merge_chk', None) is not None and self._of_merge_chk.isChecked():
                of_chain = self._degree2_chain_edges(u, v)
            else:
                of_chain = [(u, v)]
            coords_xy, mean_radius_px, _was_rev = _get_chain_coords_local(
                self.G, of_chain, margin_px=None)   # minimal junction trim
            # Also grab the UNtrimmed chain so the figure can show exactly how
            # much centerline was cropped off the ends (vs. kept for analysis).
            coords_full_xy, _, _ = _get_chain_coords_local(
                self.G, of_chain, margin_px=0.0)
            if len(of_chain) > 1:
                print(f"  merged degree-2 chain: {len(of_chain)} segments "
                      f"({of_chain[0][0]} … {of_chain[-1][1]})")
            if coords_xy is None or len(coords_xy) < 5:
                self._info_box.setPlainText(
                    f"Vessel ({u}, {v}) too short for OF "
                    f"(N={0 if coords_xy is None else len(coords_xy)}).")
                self._run_of_btn.setEnabled(True)
                return

            # 2. Load full-res video, percentile-normalise to uint8.
            print("\n  ── Optical-flow re-analysis (read-only) ──")
            print(f"  edge ({u}, {v}), tile {tile_id}, "
                  f"video {video_path}")
            # No frame cap → matches batch's default (max_frames=None loads
            # ALL raw frames, then cut_before_fade trims the fade-out).
            # Fresh-OF and batch now see identical stacks.
            stack = load_tiff_stack(video_path, max_frames=None)
            stack, _cut_at = cut_before_fade(stack, verbose=False)
            # Pass the float32 stack straight through — do NOT pre-clip to
            # uint8.  The batch does the same, and compute_of_profile_and_Q
            # then applies its bg_percentile background subtraction
            # (of_profile L29-30).  Pre-converting to uint8 here made the
            # function see uint8 and SKIP the background subtraction (L31) →
            # OF tracked the static background instead of the moving speckle,
            # producing the "awful" fresh-PIV result.  Keep it float.
            stack = np.asarray(stack, dtype=np.float32)
            T, H, W = stack.shape
            print(f"  compute source: {T} frames, {H}×{W}, "
                  f"radius (mosaic) = {mean_radius_px:.1f} px")

            # Downsample factor is used ONLY for the px→µm unit conversion
            # below.  Geometry is handled natively by mosaic_to_tile_coords
            # (via video_w/tile_width), so it is correct at any resolution.
            ds_y = float(H) / float(TILE_RAW_HEIGHT) if H else 1.0
            ds_x = float(W) / float(TILE_RAW_WIDTH) if W else 1.0

            # 3. Convert mosaic Cartesian coords → tile pixel using the SAME
            #    batch transforms (mosaic_to_tile_coords + mosaic_radius_to_tile
            #    with affine re-referencing), so a fresh recompute reproduces
            #    the stored Q(t) exactly.  The earlier inline (coord−translate)/
            #    scale did NOT apply the affine re-reference or the
            #    video_w/tile_width scaling → the centerline was misplaced and
            #    OF integrated the wrong pixels (the "awful" result).
            # Reuse the batch analyzer's geom object — same source of truth
            # so a fresh recompute here matches the stored batch measurement
            # exactly.  Cached on the viewer after first build.
            if getattr(self, '_of_geom', None) is None:
                cfg = StageConfig.load(self.graph_path.parents[1])
                all_shapes = {tid: shp for tid in cfg.discover_raw_tile_ids()
                              for shp in [tile_video_shape(cfg, tid)]
                              if shp is not None}
                self._of_geom = load_tile_geometry(cfg, video_shapes=all_shapes)
            geom = self._of_geom
            centerline_rc = graph_to_tile_batch(coords_xy, geom, int(tile_id))
            # Untrimmed full-vessel CL in the same tile frame (for the figure's
            # "how much was cropped" overlay).  Not in-bounds-trimmed.
            try:
                centerline_full_rc = graph_to_tile_batch(
                    coords_full_xy, geom, int(tile_id))
            except Exception:
                centerline_full_rc = centerline_rc
            # Trim to largest in-bounds contiguous segment
            margin = 2
            ib = ((centerline_rc[:, 0] >= margin) & (centerline_rc[:, 0] <= H - margin) &
                  (centerline_rc[:, 1] >= margin) & (centerline_rc[:, 1] <= W - margin))
            if np.any(ib) and not np.all(ib):
                changes = np.diff(ib.astype(int))
                starts = list(np.where(changes == 1)[0] + 1)
                ends = list(np.where(changes == -1)[0] + 1)
                if ib[0]:  starts = [0] + starts
                if ib[-1]: ends = ends + [len(ib)]
                if starts and ends:
                    n = min(len(starts), len(ends))
                    best = int(np.argmax(np.array(ends[:n]) - np.array(starts[:n])))
                    centerline_rc = centerline_rc[starts[best]:ends[best]]
            r_tile = float(graph_radius_to_tile_px(
                float(mean_radius_px), geom, int(tile_id)))
            vessel_angle = float(np.arctan2(
                centerline_rc[-1, 0] - centerline_rc[0, 0],
                centerline_rc[-1, 1] - centerline_rc[0, 1]))
            print(f"  centerline tile range: rows "
                  f"[{centerline_rc[:, 0].min():.0f}, {centerline_rc[:, 0].max():.0f}], cols "
                  f"[{centerline_rc[:, 1].min():.0f}, {centerline_rc[:, 1].max():.0f}]")
            print(f"  radius (tile)  = {r_tile:.1f} px  "
                  f"(θ = {np.degrees(vessel_angle):.0f}°)")

            # 4. Run the unified OF pipeline — same call as the editing
            # viewer's Analyze Mode.
            edge_data = self.G.edges[u, v]
            # v4: f0 lives per-tile in measurements_piv[tile]['harm']['f0_hz'],
            # not on the edge.  Reading the dead flat 'f0_hz' returned None and
            # silently fell back to 2.5 Hz — which mis-set the cycle-averaging
            # period and the harmonic fit (flat waveform, SNR≈0).  Use the tile.
            f0 = self._tile_f0_hz(edge_data, tile_id)
            # Farneback params match the frozen calibration
            # (memory/of_calibration_frozen.md): fixed winsize=3, dt=1.
            # Batch pipeline uses these; keeping fresh OF aligned so a
            # fresh recompute reproduces stored Q(t) for the same edge.
            fb_ws = 3
            unified = compute_of_profile_and_Q(
                stack, centerline_rc, r_tile,
                bg_percentile=10,
                fb_winsize=fb_ws, fb_levels=1,
                fb_iterations=3,
                fb_poly_n=3, fb_poly_sigma=0.7,
                # dt=1: reverted from dt=3 (2026-08-24-verified aliasing on
                # fast A vessels — v~600-1000 µm/s hits 4-6 px/frame at dt=3,
                # well beyond Farneback winsize=3 capture, flat-tops the peak).
                # dt=1 is unbiased at all speeds; a bit noisier for slow venous.
                dt=1, n_bins=30, fit_fraction=0.75,
                refine_centerline=(getattr(self, '_of_refine_chk', None) is None
                                   or self._of_refine_chk.isChecked()),
                use_flow_tangent=True,
                tangent_smooth_sigma=5.0,
                f0_hz=f0, frame_dt_s=FRAME_DT_S,
                return_diagnostics=True,   # for the "Show OF pipeline" button
            )
            # DEBUG: confirm the σ=4 centerline smoothing is reaching fresh OF.
            _clu = np.asarray(unified.get('cl_used')) if unified.get('cl_used') is not None else None
            _rgh = (float(np.mean(np.abs(_clu[2:] - 2 * _clu[1:-1] + _clu[:-2])))
                    if _clu is not None and len(_clu) > 3 else -1.0)
            print(f"[fresh OF] cl_used roughness = {_rgh:.3f} px  "
                  f"(smoothed σ=4 ≈ <1 ; jagged σ=0 ≈ >1.5)")
            elapsed = _time.time() - t0
        except Exception as e:
            import traceback; traceback.print_exc()
            self._info_box.setPlainText(
                f"Optical flow failed: {type(e).__name__}: {e}")
            self._run_of_btn.setEnabled(True)
            return

        # 5. Build the of_result dict in the shape _analyze_of_display
        #    expects — mirrors the editing viewer's adapter at
        #    pertile/viewer/mosaic/_kirchhoff.py:2870-2937.
        #
        # If the bundle ships pre-downsampled videos (e.g. ds_x = 0.5
        # for a 2× spatial downsample) one displayed pixel covers
        # (1 / ds) × PX_SIZE_UM real micrometers, so use the effective
        # pixel size for the unit conversion.  At ds = 1 (full-res)
        # this collapses to the original formula.
        ds_avg = 0.5 * (ds_x + ds_y) if (ds_x and ds_y) else 1.0
        PX_SIZE_EFF = float(PX_SIZE_UM) / ds_avg
        conv = PX_SIZE_EFF ** 3 / FRAME_DT_S * 1e-6  # px²/frame → nL/s
        v_conv = PX_SIZE_EFF / FRAME_DT_S            # px/frame  → µm/s
        R_um = r_tile * PX_SIZE_EFF

        # ── Segmentation-CL re-bin of the SAME OF field (no extra Farneback):
        # this reproduces the batch's Q_pois_seg — the canonical default since
        # the KCL bake-off (2026-07; raw seg conserves mass better than the
        # ridge-refined CL or the r(R)-calibrated value).  We report seg as the
        # headline Q and rescale the (ridge) Poiseuille waveform to the seg
        # mean, so the fresh overlay matches the stored Q(t) — which the batch
        # now builds the same way (repoint block in batch_analyze_mosaic_piv).
        # The ridge value is kept and still shown for reference.
        Q_pois_ridge_px = float(unified.get('Q_pois_mean', np.nan))
        Q_pois_seg_px = Q_pois_ridge_px
        pois_seg_factor = 1.0
        try:
            from ..analyze.profile import _rebin_profile
            _seg = _rebin_profile(unified, centerline_rc, r_tile,
                                  n_bins=30, fit_fraction=0.75)
            _sv = float(_seg.get('Q_pois_mean', np.nan))
            if (np.isfinite(_sv) and np.isfinite(Q_pois_ridge_px)
                    and abs(Q_pois_ridge_px) > 1e-9):
                Q_pois_seg_px = _sv
                pois_seg_factor = abs(_sv) / abs(Q_pois_ridge_px)
        except Exception:
            import traceback; traceback.print_exc()

        # Match the BATCH canonical scale target (mosaic_piv:_analyze_vessel):
        # Kloosterman → Q_pois_free → seg.  Kloosterman is the citable,
        # most-defensible method, so fresh OF's headline Q and rescaled
        # waveform reproduce the stored batch Q(t) instead of the old
        # seg-Poiseuille default.  (Overrides pois_seg_factor computed above.)
        _Q_kl_px = float(unified.get('Q_kloosterman', np.nan))
        _v0_free = float(unified.get('v0_free', np.nan))
        _R_free_px = float(unified.get('R_fit_free', np.nan))
        _Q_pf_px = (abs(_v0_free) * np.pi * _R_free_px ** 2 / 2.0
                    if (np.isfinite(_v0_free) and np.isfinite(_R_free_px)
                        and _R_free_px > 0) else np.nan)
        # Kloosterman → bounded mask-R Poiseuille (seg) → railing-prone free-R.
        if np.isfinite(_Q_kl_px):
            _Q_scale_px = _Q_kl_px; self._of_best_key = 'kl'
        elif np.isfinite(Q_pois_seg_px):
            _Q_scale_px = Q_pois_seg_px; self._of_best_key = 'pois_seg'
        elif np.isfinite(_Q_pf_px):
            _Q_scale_px = _Q_pf_px; self._of_best_key = 'pois_free'
        else:
            _Q_scale_px = Q_pois_seg_px; self._of_best_key = 'pois_seg'
        if np.isfinite(_Q_scale_px) and abs(Q_pois_ridge_px) > 1e-9:
            pois_seg_factor = abs(_Q_scale_px) / abs(Q_pois_ridge_px)

        # Stash the full result for the "Show OF pipeline" button (4-figure
        # diagnostic).  Grab this tile's stored Q(t) for the fresh-vs-stored panel.
        _sQ = None
        for _mm in (self.G.edges[u, v].get('measurements_piv') or []):
            if _mm.get('tile_id') == int(tile_id):
                _sQ = _mm.get('Q_t'); break
        self._last_of_pipeline = dict(
            res=unified, frame_dt_s=FRAME_DT_S, v_conv=v_conv, q_conv=conv,
            stored_Qt=_sQ, f0_hz=f0, title=f"edge ({u}, {v}) tile {tile_id}",
            # Which vessel/tile this result is for — lets the "Diagnose
            # vessel quality" button tell whether the stashed OF matches the
            # currently-selected edge (and re-run fresh OF if it's stale).
            edge=(int(u), int(v)), tile_id=int(tile_id),
            # Canonical-Q scale + key so the pipeline figure can rebuild the
            # reported Q(t) waveform straight from `unified` (one data path).
            pois_seg_factor=float(pois_seg_factor),
            best_key=getattr(self, '_of_best_key', 'kl'),
            # Stash the raw stack + geometry so the PIV comparison button can
            # rerun on identical inputs without reloading video from disk.
            frames=stack, centerline_rc=centerline_rc,
            centerline_full_rc=centerline_full_rc, r_tile=r_tile)
        if hasattr(self, '_of_pipeline_btn'):
            self._of_pipeline_btn.setEnabled(True)
        Q_plug_nLs = unified['Q_plug_mean'] * conv
        Q_pois_nLs = unified['Q_pois_mean'] * conv       # ridge CL (reference)
        Q_pois_seg_nLs = Q_pois_seg_px * conv            # seg CL = canonical default
        v0_fix_ums = unified['v0_fixed'] * v_conv
        v0_free_ums = unified.get('v0_free', np.nan) * v_conv
        R_fit_free = unified.get('R_fit_free', r_tile)
        Q_pois_free_nLs = (unified.get('v0_free', np.nan)
                            * np.pi * R_fit_free ** 2 / 2.0 * conv)
        v_mean_ums = Q_plug_nLs * 1e6 / (np.pi * R_um ** 2)
        vf = unified.get('vfield_mean')
        roi_offset = unified.get('roi_offset', (0, 0))
        r0_roi, c0_roi = int(roi_offset[0]), int(roi_offset[1])
        if vf is not None:
            roi_h, roi_w = vf.shape[:2]
        else:
            roi_h, roi_w = H, W
            r0_roi = c0_roi = 0
        rr_in = unified['rr_in']
        cc_in = unified['cc_in']
        mask_roi = np.zeros((roi_h, roi_w), dtype=bool)
        if len(rr_in) > 0:
            mask_roi[rr_in, cc_in] = True
        Q_pois_orig_nLs = unified.get(
            'Q_pois_orig_mean', np.nan) * conv
        v0_orig_ums = unified.get('v0_orig', np.nan) * v_conv

        of_result = dict(
            Q_mean=Q_plug_nLs,
            Q_poiseuille=Q_pois_seg_nLs,   # canonical default = seg CL
            Q_poiseuille_ridge=Q_pois_nLs,
            Q_pois_orig=Q_pois_orig_nLs,
            Q_pois_free=Q_pois_free_nLs,
            v_mean=v_mean_ums,
            v0_fit=v0_fix_ums,
            v0_free=v0_free_ums,
            v0_orig=v0_orig_ums,
            R_stored_px=r_tile,
            R_flow_px=R_fit_free,
            R_flow_um=R_fit_free * PX_SIZE_UM,
            delta_px=unified.get('cl_shift', 0.0),
            best_side='center',
            fit_left=None, fit_right=None,
            c_floor=np.nan,
            profile_r=unified['bin_centers'],
            profile_v=np.abs(unified['profile_v']) * v_conv,
            profile_v_orig=np.abs(unified.get('profile_v_orig',
                np.full_like(unified['profile_v'], np.nan))) * v_conv,
            profile_r_local=None, profile_v_local=None,
            Q_t_of_mean=unified['Q_plug_t'] * conv,
            # ridge per-frame waveform rescaled to the seg mean (canonical
            # default), so this matches the stored Q(t) built by the batch.
            Q_t_of_pois=unified['Q_pois_t'] * conv * pois_seg_factor,
            Q_t_of_local=None,
            Q_local=np.nan,
            v_field_mean=(vf * v_conv if vf is not None else None),
            roi_bounds=(r0_roi, r0_roi + roi_h, c0_roi, c0_roi + roi_w),
            vessel_mask_roi=mask_roi,
            wide_mask_roi=mask_roi,
            cl_shift=unified.get('cl_shift', 0.0),
            cl_used=unified.get('cl_used'),
            # Initial (pre-refinement) tile-frame centerline the OF pipeline
            # received.  Kept so the diagnostic can plot both CLs on the
            # mean-field panel and expose how much the refinement moved.
            cl_initial=np.asarray(centerline_rc, dtype=float),
            v0_deconv=unified.get('v0_deconv', np.nan) * v_conv,
            R_deconv=unified.get('R_deconv', r_tile),
            sigma_of_fit=unified.get('sigma_of_fit', np.nan),
        )

        # 6. Curved ROI overlay on the mosaic — DISABLED in the plexusflow
        #    port: it used pertile's simple scale+translate tile model
        #    (`_of_tile_geometry` → `batch_analyze_mosaic.parse_tile_positions`),
        #    which doesn't exist here — this port uses the affine `_of_geom`.
        #    The walkthrough figure's ROI-locator panel (full tile + green crop
        #    box, mosaic-oriented) now serves the same "where is the crop"
        #    purpose, so this napari ribbon is not re-implemented.

        # 7. Info-box summary (matches what shows in the editing viewer's
        #    console, condensed).
        self._info_box.setPlainText(
            f"Optical flow — edge ({u}, {v}), tile {tile_id}\n"
            f"  vessel            = {len(of_chain)} segment(s) merged"
            f"{' [whole degree-2 vessel]' if len(of_chain) > 1 else ' [single segment]'}\n"
            f"  source            = {video_path}\n"
            f"  compute frames    = {T} × {H} × {W}\n"
            f"  Q_pois (seg CL)   = {abs(Q_pois_seg_nLs):.4f} nL/s  [canonical default]\n"
            f"  Q_pois (ridge CL) = {abs(Q_pois_nLs):.4f} nL/s  (×{pois_seg_factor:.3f} → seg)\n"
            f"  Q_plug (ridge CL) = {abs(Q_plug_nLs):.4f} nL/s\n"
            f"  v̄                 = {abs(v_mean_ums):.0f} µm/s\n"
            f"  R (tile)          = {r_tile:.2f} px  "
            f"({R_um:.2f} µm)\n"
            f"  CL shift          = {unified.get('cl_shift', 0.0):+.1f} px\n"
            f"  computed in {elapsed:.1f}s   "
            f"(read-only: graph not modified)")

        # 7. Render the comprehensive stage-by-stage OF pipeline walkthrough
        #    (single figure, one data path off `unified`/`_last_of_pipeline`).
        #    Replaces the old 4-panel `_render_of_diagnostic` popup — that
        #    method is retained but unused, in case a minimal view is wanted.
        try:
            self._render_of_pipeline(self._last_of_pipeline)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._info_box.setPlainText(
                self._info_box.toPlainText()
                + f"\n\nPipeline figure failed: "
                f"{type(e).__name__}: {e}")
        self._run_of_btn.setEnabled(True)


    def _tile_f0_hz(self, edge_data, tile_id, fallback=2.5):
        """Heart-rate f0 (Hz) for a given tile's measurement on this edge.

        v4 stores f0 per measurement in ``measurements_piv[tile]['harm']
        ['f0_hz']`` — NOT as a flat edge field.  Falls back to the tile-f0
        graph metadata, then any other measurement's f0, then a sane yolk-sac
        default, so the cycle-averaging period and harmonic fit are never
        driven by a stale 2.5 Hz."""
        def _ok(x):
            try:
                return x is not None and np.isfinite(float(x)) and float(x) > 0
            except (TypeError, ValueError):
                return False
        meas = edge_data.get('measurements_piv') or []
        # 1. this tile's own measurement
        for mm in meas:
            if mm.get('tile_id') == int(tile_id):
                f0 = (mm.get('harm') or {}).get('f0_hz')
                if _ok(f0):
                    return float(f0)
                break
        # 2. graph-level per-tile f0 metadata — try the canonical key
        # `tile_f0_piv` (what batch writes) first, then the legacy
        # `tile_f0s` for backwards compat.
        for meta_key in ('tile_f0_piv', 'tile_f0s'):
            tf = getattr(self.G, 'graph', {}).get(meta_key) or {}
            for key in (int(tile_id), str(tile_id)):
                if _ok(tf.get(key)):
                    return float(tf[key])
        # 3. any measurement on this edge with a usable f0
        for mm in meas:
            f0 = (mm.get('harm') or {}).get('f0_hz')
            if _ok(f0):
                return float(f0)
        # 4. legacy flat field, then default
        f0 = edge_data.get('f0_hz')
        return float(f0) if _ok(f0) else float(fallback)


    def _show_of_region_overlay(self, of_result, tile_id, u, v,
                                  sx, sy, trans_x, trans_y_disp):
        """Render the OF integration mask as a curved orange ribbon on
        the mosaic.  Replaces any previous OF overlay.

        Pipeline:
          1.  Extract mask boundary contour(s) in ROI-pixel coords
              via skimage.measure.find_contours.
          2.  ROI → tile pixel (add roi_offset).
          3.  Tile pixel → mosaic display (multiply by sx/sy, add tile
              translates).  Inverse of the forward transform used to
              build centerline_rc for compute_of_profile_and_Q.
          4.  Add as a napari Shapes layer (polygon, orange fill +
              outline).  Mosaic display coords are (row, col) with
              row 0 at the top — matches every other Shapes layer in
              this viewer.

        Persists until the next OF run replaces it or
        _tear_down_of_region_overlay clears it explicitly."""
        try:
            from skimage.measure import find_contours
        except ImportError:
            print("  (skimage missing — skipping ROI overlay)")
            return
        mask = of_result.get('vessel_mask_roi')
        if mask is None or not mask.any():
            return
        roi_bounds = of_result.get('roi_bounds')
        if roi_bounds is None:
            return
        r0_roi, _r1_roi, c0_roi, _c1_roi = roi_bounds
        # find_contours returns a list of (N, 2) arrays in (row, col)
        # ROI coords.  Each ring becomes one polygon — keep them all so
        # vessels with internal holes (rare) draw correctly.
        contours = find_contours(mask.astype(float), level=0.5)
        if not contours:
            return
        polys = []
        H_mosaic = self.mosaic_height
        for c in contours:
            if len(c) < 3:
                continue
            i_roi = c[:, 0]
            j_roi = c[:, 1]
            # ROI → tile pixel
            i_tile = i_roi + r0_roi
            j_tile = j_roi + c0_roi
            # Tile pixel → mosaic display (inverse of the j_tile / i_tile
            # formulae used when building centerline_rc).  display_row
            # is the napari row index (row 0 at the top).
            display_row = i_tile * sy + trans_y_disp
            display_col = j_tile * sx + trans_x
            # napari Shapes wants (row, col) — same as display.  No
            # mosaic-height flip needed here because we never went
            # through the Cartesian-y representation.
            poly = np.column_stack([display_row, display_col])
            polys.append(poly)
        # Replace any prior overlay before adding the new one.
        self._tear_down_of_region_overlay(suppress_button=True)
        if not polys:
            return
        _prev_active = self.viewer.layers.selection.active
        self._of_region_layer = self.viewer.add_shapes(
            polys,
            shape_type='polygon',
            edge_color='#ff8800',
            face_color='#ff8800',
            edge_width=2.5,
            opacity=0.32,
            name=f'OF region — edge ({u},{v}), tile {tile_id}',
        )
        # Don't let the overlay capture mouse events — clicks must still
        # reach the edges layer for selection / inspection.
        try:
            self._of_region_layer.mouse_pan = False
            self._of_region_layer.mouse_zoom = False
        except Exception:
            pass
        if _prev_active is not None:
            try:
                self.viewer.layers.selection.active = _prev_active
            except Exception:
                pass
        try:
            self._clear_of_region_btn.setEnabled(True)
        except AttributeError:
            pass


    def _tear_down_of_region_overlay(self, suppress_button: bool = False):
        """Remove the OF region Shapes layer if present.  `suppress_button`
        skips the Clear-button disable (used internally by the show path
        when replacing one overlay with another)."""
        lyr = getattr(self, '_of_region_layer', None)
        if lyr is not None:
            try:
                self.viewer.layers.remove(lyr)
            except (KeyError, ValueError):
                pass
            self._of_region_layer = None
        if not suppress_button:
            try:
                self._clear_of_region_btn.setEnabled(False)
            except AttributeError:
                pass
