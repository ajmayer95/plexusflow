"""Per-vessel likelihoods for the harmonic comparison (scales profiled out by 1-D minimisation).

nll_vm / nll_t: relative weights given (κ_e ∝ w_e, scale ∝ s_e), one overall scale profiled.
nll_vm_var / nll_t_var: per-vessel variance V_e = V_beat,e + (τ c_e)² — the measurement's own waveform repeatability plus a
systematic term proportional to its geometric reliability c; τ profiled.  von Mises κ from V by the circular-variance
relation A1(κ) = I1(κ)/I0(κ) = e^{−V/2}, so very noisy phases (V large, κ → 0) drop out smoothly.
"""
from __future__ import annotations
import numpy as np
from scipy.special import i0e, gammaln
from scipy.optimize import minimize_scalar

NU = 4.0        # Student-t degrees of freedom for the log-pulsatility ratio


def nll_vm(dphi, w):
    """von Mises NLL of phase errors with κ_e = κ0·w_e, minimised over κ0."""
    def f(lk):
        k = np.exp(lk) * w
        return np.sum(-k * np.cos(dphi) + np.log(2 * np.pi * i0e(k)) + k)     # log I0(k) = log i0e(k) + k
    return minimize_scalar(f, bounds=(-8, 6), method="bounded").fun


def nll_t(r, s):
    """Student-t (ν = NU) NLL of r with scale s0·s_e, minimised over s0."""
    def f(ls):
        sc = np.exp(ls) * s; z = r / sc
        return np.sum(-gammaln((NU + 1) / 2) + gammaln(NU / 2) + 0.5 * np.log(NU * np.pi) + np.log(sc)
                      + (NU + 1) / 2 * np.log1p(z ** 2 / NU))
    return minimize_scalar(f, bounds=(-8, 4), method="bounded").fun


def a1inv(R):
    """Inverse of A1(κ) = I1(κ)/I0(κ) (Best & Fisher 1981 approximation)."""
    R = np.clip(R, 1e-6, 1 - 1e-9)
    return np.where(R < 0.53, 2 * R + R ** 3 + 5 * R ** 5 / 6, np.where(R < 0.85, -0.4 + 1.39 * R + 0.43 / (1 - R),
                                                                         1 / (R ** 3 - 4 * R ** 2 + 3 * R)))


def kappa_of_var(V):
    """von Mises concentration with circular variance matching a phase variance V (rad²)."""
    return a1inv(np.exp(-np.asarray(V) / 2))


def nll_vm_var(dphi, V_beat, c):
    """von Mises NLL, κ_e = κ(V_beat,e + (τ c_e)²), minimised over τ.  Returns (NLL, κ_e at the optimum)."""
    def f(lt):
        k = kappa_of_var(V_beat + (np.exp(lt) * c) ** 2)
        return np.sum(-k * np.cos(dphi) + np.log(2 * np.pi * i0e(k)) + k)
    r = minimize_scalar(f, bounds=(-8, 3), method="bounded")
    return r.fun, kappa_of_var(V_beat + (np.exp(r.x) * c) ** 2)


def nll_t_var(r, V_beat, c):
    """Student-t (ν = NU) NLL of r with scale √(V_beat,e + (τ c_e)²), minimised over τ."""
    def f(lt):
        sc = np.sqrt(V_beat + (np.exp(lt) * c) ** 2); z = r / sc
        return np.sum(-gammaln((NU + 1) / 2) + gammaln(NU / 2) + 0.5 * np.log(NU * np.pi) + np.log(sc)
                      + (NU + 1) / 2 * np.log1p(z ** 2 / NU))
    return minimize_scalar(f, bounds=(-8, 3), method="bounded").fun
