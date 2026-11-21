"""Layered packed bed: grid, per-cell sorbent data and the shared rate kernel.

Origin: the kernel generalizes psa-viz ``psaviz/column.py`` (``ColumnModel.velocities``
and ``ColumnModel.rhs``; git@github.com:L-u-k-e-Walker/psa-viz.git, commit 709af6f),
itself a compact rewrite of psa-sim's isothermal, no-pressure-drop, flow-controlled
model. Same equations, generalized to n components, layers of different sorbents
(per-cell void, bulk density, isotherm and LDF data), non-uniform cells and several
valves per bed end:

* Gas in cell n (superficial face velocities u, upwind face concentrations):
  eps_n dz_n dc_i/dt = F_i(face n) - F_i(face n+1) - rho_n dz_n dq_i/dt
* LDF: dq_i/dt = k_i (q*_i - q_i); on Darken sorbents k_i = k_i0 / (1 - theta).
* No momentum balance: the pressure is uniform along the bed. Face velocities come
  from the overall balance, marched cell by cell:
  u_{n+1} = u_n - vads_n - eps_n dz_n G,  vads_n = rho_n dz_n sum_i dq_i/dt / C_n,
  G = (dC/dt)/C = (u_0 - u_N - sum vads) / sum(eps dz) when both end flows are set
  by valves (or closed), G = 0 when one end is "free" (constant pressure, psa-viz's
  FEED and PURGE steps; used for the regression gate G1c).
* End molar flows are sums over the open valves at that end; inflowing gas carries
  its source's composition, outflowing gas the end cell's composition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .isotherm import R_GAS, THETA_MAX, Sorbent, loading_cells


@dataclass
class Layer:
    name: str
    sorbent: Sorbent
    height: float  # m
    void: float  # interparticle void fraction
    bulk_density: float  # kg of sorbent per m^3 of bed
    n_cells: int


class BedGrid:
    """Cells, per-cell parameters and the rate kernel of one bed (cell 0 = feed end, bottom)."""

    def __init__(self, layers, radius, temperature):
        self.layers = list(layers)
        self.radius = float(radius)
        self.area = math.pi * radius**2
        self.T = float(temperature)
        self.RT = R_GAS * self.T
        n = {lay.sorbent.n for lay in self.layers}
        if len(n) != 1:
            raise ValueError("all sorbents need the same component count")
        self.n = n.pop()
        S = max(lay.sorbent.qs.shape[1] for lay in self.layers)
        dz, eps, rho, layer_of = [], [], [], []
        qs, b, k, dk = [], [], [], []
        for li, lay in enumerate(self.layers):
            m = int(lay.n_cells)
            dz += [lay.height / m] * m
            eps += [lay.void] * m
            rho += [lay.bulk_density] * m
            layer_of += [li] * m
            s = lay.sorbent
            qs_l = np.zeros((self.n, S))
            b_l = np.zeros((self.n, S))
            qs_l[:, :s.qs.shape[1]] = s.qs
            b_l[:, :s.b.shape[1]] = s.b
            qs += [qs_l] * m
            b += [b_l] * m
            k += [s.k_ldf] * m
            dk += [bool(s.darken)] * m
        self.N = len(dz)
        self.dz = np.array(dz)
        self.eps = np.array(eps)
        self.rho = np.array(rho)
        self.layer_of = np.array(layer_of)
        self.qs = np.moveaxis(np.array(qs), 0, -1)  # (n, S, N)
        self.b = np.moveaxis(np.array(b), 0, -1)
        self.k = np.array(k).T  # (n, N)
        self.darken = np.array(dk)
        self.any_darken = bool(self.darken.any())
        qs_tot = self.qs.sum(1)  # (n, N)
        self.inv_qs_tot = np.where(qs_tot > 0, 1.0 / np.where(qs_tot > 0, qs_tot, 1.0), 0.0)
        self.epsdz = self.eps * self.dz
        self.sum_epsdz = float(self.epsdz.sum())
        self.length = float(self.dz.sum())
        self.z_faces = np.concatenate([[0.0], np.cumsum(self.dz)])
        self.z = 0.5 * (self.z_faces[1:] + self.z_faces[:-1])
        self.rho_over_eps = self.rho / self.eps
        self.volume = self.area * self.length

    # ----------------------------------------------------------------- helpers
    def equilibrium(self, c):
        """q* (n, N, K) for gas concentrations c (n, N, K)."""
        return loading_cells(c * (self.RT / 1e5), self.qs, self.b)

    def inventory(self, c, q):
        """Moles of each component held in the bed (gas + adsorbed): (n,) or (n, K)."""
        w_gas = (self.area * self.epsdz)[:, None] if c.ndim == 3 else self.area * self.epsdz
        w_ads = (self.area * self.rho * self.dz)[:, None] if c.ndim == 3 else self.area * self.rho * self.dz
        return (w_gas * c).sum(1) + (w_ads * q).sum(1)

    @staticmethod
    def end_flows(valves, y_cell):
        """Total and component molar flows INTO the bed through a list of valves.

        valves: [(F (K,), y_src (n, K) or (n, 1)), ...]; y_cell: (n, K) end-cell composition."""
        F = 0.0
        Nc = 0.0
        for Fv, ys in valves:
            F = F + Fv
            Nc = Nc + Fv * np.where(Fv >= 0.0, ys, y_cell)
        return F, Nc

    # ------------------------------------------------------------------ kernel
    def rates(self, c, q, C, bot, top):
        """Time derivatives of one bed.

        c, q: (n, N, K); C = c.sum(0) (N, K).
        bot / top: ("valves", F (K,), Ncomp (n, K)) with flows INTO the bed (mol/s),
        or ("free", y_in (n, K)) for a constant-pressure end (at most one end).
        Returns dc, dq (n, N, K), u (N+1, K) superficial velocity (+ = up),
        and the component molar flows through the bottom and top faces (+ = up), (n, K).
        """
        N, a = self.N, self.area
        K = c.shape[2]
        qstar = self.equilibrium(c)
        if self.any_darken:
            theta = (q * self.inv_qs_tot[:, :, None]).sum(0)  # (N, K)
            fac = np.where(self.darken[:, None], 1.0 / (1.0 - np.clip(theta, 0.0, THETA_MAX)), 1.0)
            dq = self.k[:, :, None] * fac[None] * (qstar - q)
        else:
            dq = self.k[:, :, None] * (qstar - q)
        vads = (self.rho * self.dz)[:, None] * dq.sum(0) / C  # (N, K) m/s
        u = np.empty((N + 1, K))
        if bot[0] == "valves" and top[0] == "valves":
            u0 = bot[1] / (a * C[0])
            uN = -top[1] / (a * C[-1])
            G = (u0 - uN - vads.sum(0)) / self.sum_epsdz
            u[0] = u0
            u[1:] = u0 - np.cumsum(vads + self.epsdz[:, None] * G, axis=0)
            u[N] = uN
        elif top[0] == "free":
            u[0] = bot[1] / (a * C[0])
            u[1:] = u[0] - np.cumsum(vads, axis=0)
        elif bot[0] == "free":
            u[N] = -top[1] / (a * C[-1])
            u[:N] = u[N] + np.cumsum(vads[::-1], axis=0)[::-1]
        else:
            raise ValueError("bad end specification")
        flux = np.empty((self.n, N + 1, K))
        flux[:, 1:N] = u[1:N] * np.where(u[1:N] >= 0.0, c[:, :N - 1], c[:, 1:])
        if bot[0] == "valves":
            flux[:, 0] = bot[2] / a
        else:
            flux[:, 0] = u[0] * np.where(u[0] >= 0.0, bot[1] * C[0], c[:, 0])
        if top[0] == "valves":
            flux[:, N] = -top[2] / a
        else:
            flux[:, N] = u[N] * np.where(u[N] >= 0.0, c[:, N - 1], top[1] * C[N - 1])
        dc = (flux[:, :-1] - flux[:, 1:]) / self.epsdz[:, None] - self.rho_over_eps[:, None] * dq
        return dc, dq, u, a * flux[:, 0], a * flux[:, N]
