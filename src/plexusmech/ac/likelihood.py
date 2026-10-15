"""Per-vessel likelihoods for the harmonic comparison (scales profiled out by 1-D minimisation)."""
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
