"""Multi-site extended Langmuir equilibrium and LDF (+ Darken) kinetics for n components.

Generalizes psa-viz ``psaviz/isotherm.py`` (binary, single-site) to n components and
S sites per sorbent:

    q*_i = sum_s qs[i, s] b[i, s] p_i / (1 + sum_j b[j, s] p_j)      p in bar, q in mol/kg

A species that does not use a site has qs = b = 0 there. One site = the classical
extended Langmuir model.

Kinetics (per sorbent): dq_i/dt = k_i (q*_i - q_i). With ``darken`` on (the CMS, user
decision 2026-11-20) the coefficient rises with loading as in the Darken relation for
a Langmuir solid, k_i = k_i0 / (1 - theta), theta = sum_i q_i / qs_i (summed over the
sorbent's sites), capped at THETA_MAX.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

R_GAS = 8.314462618  # J/(mol K)
THETA_MAX = 0.95


@dataclass
class Sorbent:
    """Equilibrium and kinetic parameters of one sorbent at the (isothermal) bed temperature."""

    name: str
    qs: np.ndarray  # (n, S) mol/kg
    b: np.ndarray  # (n, S) 1/bar
    k_ldf: np.ndarray  # (n,) 1/s; zero-loading value when darken is on
    darken: bool = False
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.qs = np.atleast_2d(np.asarray(self.qs, float))
        self.b = np.atleast_2d(np.asarray(self.b, float))
        self.k_ldf = np.asarray(self.k_ldf, float)
        if self.qs.shape != self.b.shape or self.qs.shape[0] != self.k_ldf.size:
            raise ValueError(f"{self.name}: inconsistent parameter shapes")

    @property
    def n(self):
        return self.qs.shape[0]

    def loading(self, p_bar):
        """Equilibrium loadings for partial pressures p (n, ...) in bar; returns (n, ...)."""
        p = np.maximum(np.asarray(p_bar, float), 0.0)
        ext = (slice(None), slice(None)) + (None,) * (p.ndim - 1)
        bp = self.b[ext] * p[:, None]  # (n, S, ...)
        return (self.qs[ext] * bp / (1.0 + bp.sum(0))[None]).sum(1)

    def qs_total(self):
        return self.qs.sum(1)


def loading_cells(p, qs, b):
    """Vectorized equilibrium for per-cell parameters.

    p (n, N, K) bar; qs, b (n, S, N). Returns q* (n, N, K)."""
    p = np.maximum(p, 0.0)
    bp = b[:, :, :, None] * p[:, None, :, :]  # (n, S, N, K)
    den = 1.0 + bp.sum(0)  # (S, N, K)
    return (qs[:, :, :, None] * bp / den[None]).sum(1)
