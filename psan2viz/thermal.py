"""Heat of adsorption and desorption from the isothermal solution (user, 2026-11-22: thermal plot).

The model is isothermal (user decision), so temperature is not simulated. What follows from the solution is
the heat released when the sorbent takes gas up and the heat taken up when it releases it: for each bed,
    Q(t) = sum over cells and species of  dH_i(layer) * m_cell * dq_i/dt   (W; > 0 released, < 0 taken up)
with dq/dt the solver's own uptake rates (LDF with the Darken factor). In the isothermal model this heat is
assumed to be exchanged with the bed and the column wall. The adiabatic estimate (how far an uncooled bed
would swing) is the range of the cumulative heat over the cycle divided by the sorbent heat capacity.

Isosteric heats (positive magnitudes, J/mol), with sources (also in docs/INPUTS.md):
  CMS      O2, N2  15 kJ/mol  assumed constant from M21d Fig. 3.4-1 (Clausius-Clapeyron on the CT-350 isotherms:
                              <= 17 kJ/mol at minimum loading, falling with loading)
           Ar      15 kJ/mol  = O2 (as its isotherm)
           CO2     25 kJ/mol  assumed (typical for CO2 on carbons; a trace here)
           H2O     44 kJ/mol  assumed = heat of vaporization (a trace on the CMS)
  alumina  N2, O2, Ar 15 kJ/mol  the model's own assumption (alumina isotherm moved to 293 K with 15 kJ/mol)
           CO2     33 kJ/mol  mean of the fitted site heats 39.5 and 27.0 kJ/mol (ISODB, 273/303 K)
           H2O     44.2 kJ/mol heat of vaporization at 20 degC (IF97), implied by the model's assumption that the
                              loading depends on RH only (then the isosteric heat equals the heat of vaporization)
Sorbent heat capacity 880 J/(kg K): M21d (CMS); used for the alumina as well (assumed).
"""
from __future__ import annotations

import numpy as np

HEATS = {  # J/mol, order N2, O2, Ar, CO2, H2O
    "CMS": np.array([15e3, 15e3, 15e3, 25e3, 44e3]),
    "alumina": np.array([15e3, 15e3, 15e3, 33e3, 44.2e3]),
}
CP_SORBENT = 880.0  # J/(kg K), M21d


def _cell_heats(pl):
    g = pl.g
    names = [lay.name for lay in g.layers]
    return np.stack([HEATS[names[i]] for i in g.layer_of], axis=1)  # (n, N)


def heat_rates(pl, x):
    """Heat released (> 0) or taken up (< 0) by each bed's sorbent at state x, W: [bed 1, bed 2]."""
    g = pl.g
    dH = _cell_heats(pl)
    m_cell = g.rho * g.area * g.dz  # kg of sorbent per cell
    zf = (np.zeros(1), np.zeros((pl.n, 1)))
    out = []
    for k in (0, 1):
        c, q = pl.bed(x, k)
        _, dq, *_ = g.rates(c[:, :, None], q[:, :, None], c.sum(0)[:, None], ("valves", *zf), ("valves", *zf))
        out.append(float(np.sum(dH * m_cell[None, :] * dq[:, :, 0])))
    return out


def sorbent_heat_capacity(pl):
    """J/K of sorbent in one bed."""
    g = pl.g
    return float(np.sum(g.rho * g.area * g.dz) * CP_SORBENT)
