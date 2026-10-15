"""Robust harmonic regression + per-tile f0 estimation.

Fits ``Q(t) = a0 + Σ_k [A_k cos(2πk·f0·t) + B_k sin(2πk·f0·t)]`` for
k = 1..K using IRLS with a Huber loss, plus periodogram-based tools
(``estimate_f0_in_band``, ``consensus_f0_stacked``) for picking each
tile's fundamental frequency before the harmonic fit.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np


def _mad_sigma(x: np.ndarray) -> float:
    """Robust σ estimate via the median absolute deviation (MAD)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * (mad + 1e-12)


def _irls_weights(
    resid: np.ndarray,
    loss: str,
    sigma: float,
    huber_delta_sigma: float = 1.5,
    tukey_c_sigma: float = 4.685,
) -> np.ndarray:
    """IRLS weight vector for the requested robust loss."""
    r = resid / max(sigma, 1e-12)
    if loss == "huber":
        k = float(huber_delta_sigma)
        w = np.ones_like(r)
        m = np.abs(r) > k
        w[m] = k / (np.abs(r[m]) + 1e-12)
        return w
    if loss == "tukey":
        c = float(tukey_c_sigma)
        u = r / c
        w = np.zeros_like(u)
        m = np.abs(u) < 1
        w[m] = (1 - u[m] * u[m]) ** 2
        return w
    return np.ones_like(r)


def _design_matrix(
    t: np.ndarray, f0: float, K: int, include_dc: bool = True
) -> np.ndarray:
    """Design matrix for K-harmonic regression at fundamental f0."""
    cols = []
    if include_dc:
        cols.append(np.ones_like(t))
    for k in range(1, K + 1):
        w = 2 * np.pi * k * f0
        cols.append(np.cos(w * t))
        cols.append(np.sin(w * t))
    return np.column_stack(cols)


def estimate_f0_in_band(
    v_hat: np.ndarray,
    frame_dt: float,
    fmin_hz: float = 1.0,
    fmax_hz: float = 3.5,
    df: Optional[float] = None,
    refine: bool = True,
    return_periodogram: bool = False,
):
    """Peak of a Hanning-windowed periodogram inside [fmin, fmax], with
    optional parabolic refinement in log-power. Returns ``f0`` (Hz), or
    ``(f0, freqs, power)`` when ``return_periodogram=True``. Returns NaN
    on unusable input.
    """
    _nan = (np.nan, np.array([]), np.array([])) if return_periodogram else np.nan
    y = np.asarray(v_hat, float)
    t = np.arange(y.size, dtype=float) * float(frame_dt)
    m = np.isfinite(y)
    if m.sum() < 8:
        return _nan
    y = y[m] - np.nanmedian(y[m])
    t = t[m]
    if y.size < 8 or np.allclose(y, 0):
        return _nan
    T = t[-1] - t[0]
    if T <= 0:
        return _nan
    if df is None or df <= 0:
        df = 1.0 / max(4.0 * T, 1e-12)
    f = np.arange(max(fmin_hz, df), fmax_hz + 1e-12, df, dtype=float)
    if f.size < 3:
        return _nan
    w = np.hanning(y.size)
    yw = y * w
    Y = np.array(
        [np.abs(np.sum(yw * np.exp(-2j * np.pi * ff * t))) ** 2 for ff in f],
        float,
    )
    k = int(np.argmax(Y))
    f0 = float(f[k])
    if refine and 1 <= k <= (len(f) - 2):
        xs = f[k - 1: k + 2]
        ps = np.log(Y[k - 1: k + 2] + 1e-18)
        try:
            a, b, _c = np.polyfit(xs, ps, 2)
            if a < 0:
                f_ref = -b / (2 * a)
                if fmin_hz <= f_ref <= fmax_hz:
                    f0 = float(f_ref)
        except Exception:
            pass
    if return_periodogram:
        return f0, f, Y
    return f0


def consensus_f0_stacked(
    periodograms: list,
    fmin_hz: float = 1.0,
    fmax_hz: float = 3.5,
    refine: bool = True,
) -> float:
    """Peak-normalise each per-vessel periodogram, stack, and pick the
    tallest local maximum (avoids the 1/f red-noise spike at fmin).
    Returns NaN when no valid periodograms are supplied.
    """
    from scipy.signal import find_peaks

    valid = [(fr, pw) for fr, pw in periodograms
             if len(fr) > 0 and len(pw) > 0 and np.any(pw > 0)]
    if not valid:
        return float("nan")
    f_ref = valid[0][0]
    stacked = np.zeros_like(f_ref, dtype=float)
    n_added = 0
    for fr, pw in valid:
        if len(fr) == len(f_ref) and np.allclose(fr, f_ref):
            pw_on_grid = pw
        else:
            pw_on_grid = np.interp(f_ref, fr, pw, left=0.0, right=0.0)
        peak = pw_on_grid.max()
        if peak > 0:
            stacked += pw_on_grid / peak
            n_added += 1
    if n_added == 0:
        return float("nan")
    stacked /= n_added
    peak_indices, properties = find_peaks(stacked, prominence=0)
    if len(peak_indices) > 0:
        best = peak_indices[np.argmax(properties['prominences'])]
        k = int(best)
    else:
        k = int(np.argmax(stacked))
    f0 = float(f_ref[k])
    if refine and 1 <= k <= (len(f_ref) - 2):
        xs = f_ref[k - 1: k + 2]
        ps = np.log(stacked[k - 1: k + 2] + 1e-18)
        try:
            a, b, _c = np.polyfit(xs, ps, 2)
            if a < 0:
                f_ref_val = -b / (2 * a)
                if fmin_hz <= f_ref_val <= fmax_hz:
                    f0 = float(f_ref_val)
        except Exception:
            pass
    return f0


def fit_harmonics(
    v_hat: np.ndarray,
    frame_dt: float,
    f0: float,
    K: int = 3,
    loss: Literal["huber", "tukey", "none"] = "huber",
    loss_params: Optional[Dict[str, float]] = None,
    include_dc: bool = True,
    snr_harm_cutoff_factor: float = 3.5,
) -> Dict[str, Any]:
    """Fit a harmonic model to a velocity/flow time series (with IRLS).

    Model: ``v(t) = a0 + Σ_k A_k cos(2πk·f0·t) + B_k sin(2πk·f0·t)``.

    Returns
    -------
    dict with keys:
      * ``a0`` — DC (mean) component
      * ``harmonics`` — list of {k, A, B, amp, phi, sigma_*, P_sig, snr_db}
      * ``signal`` / ``resid`` — fitted and residual time series
      * ``r2`` — coefficient of determination
      * ``snr_harm_fit_db`` / ``snr_ac_fit_db`` / ``snr_dc_fit_db``
      * ``snr_h1_db``, ``snr_h2_db``, ... — per-harmonic SNRs
      * ``amp1``, ``amp_rms_ac`` — first-harmonic amplitude and AC RMS
    """
    y0 = np.asarray(v_hat, float)
    T = int(y0.size)
    t0 = np.arange(T, dtype=float) * float(frame_dt)

    m_valid = np.isfinite(y0)
    n_valid = int(m_valid.sum())
    min_samples = max(8, 2 * int(K) + (1 if include_dc else 0) + 1)

    if n_valid < min_samples or not np.isfinite(f0):
        return dict(
            a0=np.nan,
            sigma_a0=np.nan,
            harmonics=[],
            signal=np.full(T, np.nan, float),
            resid=np.full(T, np.nan, float),
            r2=np.nan,
            hr_snr_db=np.nan,
            snr_harm_fit_db=np.nan,
            snr_ac_fit_db=np.nan,
            snr_dc_fit_db=np.nan,
            f0=float(f0),
            amp_rms_ac=np.nan,
            amp1=np.nan,
            sigma_amp1=np.nan,
            sigma_phase1=np.nan,
            **{f"snr_h{k}_db": np.nan for k in range(1, int(K) + 1)},
        )

    y = y0[m_valid]
    t = t0[m_valid]
    X = _design_matrix(t, f0, int(K), include_dc=bool(include_dc))

    # Initial OLS
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    sigma = _mad_sigma(resid)

    loss_params = loss_params or {}
    huber_delta_sigma = float(loss_params.get("huber_delta_sigma", 1.5))
    tukey_c_sigma = float(loss_params.get("tukey_c_sigma", 4.685))

    if loss != "none":
        for _ in range(8):
            w = _irls_weights(
                resid, loss, sigma,
                huber_delta_sigma=huber_delta_sigma,
                tukey_c_sigma=tukey_c_sigma,
            )
            W = np.diag(w)
            try:
                beta_new, *_ = np.linalg.lstsq(W @ X, W @ y, rcond=None)
            except Exception:
                break
            yhat_new = X @ beta_new
            resid_new = y - yhat_new
            sigma_new = _mad_sigma(resid_new)
            converged = np.allclose(beta_new, beta, rtol=1e-4, atol=1e-6)
            beta, yhat, resid, sigma = beta_new, yhat_new, resid_new, sigma_new
            if converged:
                break

    # R² and coarse SNR from R²
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-18
    ss_res = float(np.sum(resid ** 2))
    r2 = float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))
    eps = 1e-9
    hr_snr_db = 10.0 * np.log10(max(r2, eps) / max(1.0 - r2, eps))

    # Coefficient covariance for uncertainty propagation
    n_params = X.shape[1]
    dof = max(n_valid - n_params, 1)
    sigma_resid_sq = ss_res / dof
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        cov_beta = sigma_resid_sq * XtX_inv
    except np.linalg.LinAlgError:
        cov_beta = np.full((n_params, n_params), np.nan)

    idx0 = 0
    if include_dc:
        a0 = float(beta[0])
        sigma_a0 = float(np.sqrt(cov_beta[0, 0])) if np.isfinite(cov_beta[0, 0]) else np.nan
        idx0 = 1
    else:
        a0 = 0.0
        sigma_a0 = 0.0

    AB = beta[idx0:]
    harmonics = []
    P_sig = 0.0
    amp1 = np.nan
    sigma_amp1 = np.nan
    sigma_phase1 = np.nan

    for k in range(1, int(K) + 1):
        A = float(AB[2 * (k - 1)])
        B = float(AB[2 * (k - 1) + 1])
        amp = float(np.hypot(A, B))
        phi = float(np.arctan2(-B, A))

        idx_A = idx0 + 2 * (k - 1)
        idx_B = idx0 + 2 * (k - 1) + 1
        sigma_A = float(np.sqrt(cov_beta[idx_A, idx_A])) if np.isfinite(cov_beta[idx_A, idx_A]) else np.nan
        sigma_B = float(np.sqrt(cov_beta[idx_B, idx_B])) if np.isfinite(cov_beta[idx_B, idx_B]) else np.nan
        if amp > 1e-12 and np.isfinite(sigma_A) and np.isfinite(sigma_B):
            sigma_amp = np.sqrt((A * sigma_A) ** 2 + (B * sigma_B) ** 2) / amp
            sigma_phi = np.sqrt((B * sigma_A) ** 2 + (A * sigma_B) ** 2) / (amp ** 2)
        else:
            sigma_amp = np.nan
            sigma_phi = np.nan

        P_sig_k = 0.5 * (A * A + B * B)          # Parseval per-harmonic power
        harmonics.append(dict(k=k, A=A, B=B, amp=amp, phi=phi,
                              sigma_A=sigma_A, sigma_B=sigma_B,
                              sigma_amp=sigma_amp, sigma_phi=sigma_phi,
                              P_sig=P_sig_k))
        P_sig += P_sig_k
        if k == 1:
            amp1 = amp
            sigma_amp1 = sigma_amp
            sigma_phase1 = sigma_phi

    amp_rms_ac = float(np.sqrt(max(P_sig, 0.0)))

    # ── Band-limited SNRs (DC, AC, per-harmonic) ─────────────────────
    signal_power_fit = float(a0 * a0 + P_sig)
    signal_power_ac = float(P_sig)
    signal_power_dc = float(a0 * a0)
    fs_fit = 1.0 / float(frame_dt) if frame_dt > 0 else np.nan
    cutoff_hz = float(snr_harm_cutoff_factor) * float(f0)
    snr_harm_fit_db = snr_ac_fit_db = snr_dc_fit_db = float("nan")
    if (np.isfinite(fs_fit) and np.isfinite(cutoff_hz)
            and cutoff_hz > 0 and len(resid) >= 8):
        N_res = len(resid)
        spec_res = np.fft.rfft(resid)
        freqs_res = np.fft.rfftfreq(N_res, d=float(frame_dt))
        spec_bl = spec_res.copy()
        spec_bl[freqs_res > cutoff_hz] = 0.0
        resid_bl = np.fft.irfft(spec_bl, n=N_res)
        ms_bl = float(np.mean(resid_bl ** 2))
        if ms_bl > 0 and signal_power_fit > 0:
            snr_harm_fit_db = 10.0 * np.log10(signal_power_fit / ms_bl)
        if ms_bl > 0 and signal_power_ac > 0:
            snr_ac_fit_db = 10.0 * np.log10(signal_power_ac / ms_bl)
        if ms_bl > 0 and signal_power_dc > 0:
            snr_dc_fit_db = 10.0 * np.log10(signal_power_dc / ms_bl)
        for hd in harmonics:
            hd["snr_db"] = (10.0 * np.log10(hd["P_sig"] / ms_bl)
                             if ms_bl > 0 and hd["P_sig"] > 0
                             else float("nan"))
    else:
        for hd in harmonics:
            hd["snr_db"] = float("nan")

    signal_full = np.full(T, np.nan, float)
    resid_full = np.full(T, np.nan, float)
    signal_full[m_valid] = yhat
    resid_full[m_valid] = resid

    return dict(
        a0=a0,
        sigma_a0=sigma_a0,
        harmonics=harmonics,
        signal=signal_full,
        resid=resid_full,
        r2=r2,
        hr_snr_db=hr_snr_db,
        snr_harm_fit_db=snr_harm_fit_db,
        snr_ac_fit_db=snr_ac_fit_db,
        snr_dc_fit_db=snr_dc_fit_db,
        **{f"snr_h{hd['k']}_db": hd.get("snr_db", float("nan"))
           for hd in harmonics},
        f0=float(f0),
        amp_rms_ac=amp_rms_ac,
        amp1=amp1,
        sigma_amp1=sigma_amp1,
        sigma_phase1=sigma_phase1,
    )
