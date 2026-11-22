"""Rates and pressure-work energy of a CSS cycle, in litres and SLPM (user, 2026-11-22: no m^3).

All from the cycle metrics (Plant.cycle_metrics) and the case, so the numbers follow the physics:
  product rate      SLPM at 21.1 degC, 1 atm (the user's standard for the 30 SLPM spec)
  feed air          SLPM, the same standard; air factor = feed air / product
  pressure work     ideal isothermal compression of the feed air from 1 atm to the feed header
                    (7.5 barg): W = n_feed R T ln(p_header / p_atm), T = the model temperature (20 degC).
                    This is the thermodynamic minimum ("pressure work"); a real compressor needs more
                    (adiabatic losses, motor and drive efficiency): stated, not estimated here.
  specific energy   pressure work per litre of product (Wh per standard L)
  productivity      product SLPM per litre of CMS (both beds)
"""
from __future__ import annotations

import math

from .cases import P_ATM, STD_P, STD_T

R = 8.314462618
L_PER_MOL_STD = R * STD_T / STD_P * 1000.0  # 24.145 L/mol at 21.1 degC, 1 atm


def performance(met, case):
    a = case.case_args
    t_cyc = float(met["t_cycle_s"])
    feed_mol_s = float(sum(met["feed_mol"])) / t_cyc
    prod_mol_s = float(sum(met["product_mol"])) / t_cyc
    T = float(case.grid.T)
    w_iso = feed_mol_s * R * T * math.log(float(case.p_feed) / P_ATM)  # W
    prod_slpm = prod_mol_s * 60.0 * L_PER_MOL_STD
    feed_slpm = feed_mol_s * 60.0 * L_PER_MOL_STD
    v_cms_l = 2.0 * math.pi * a["radius"] ** 2 * a["h_cms"] * 1000.0  # both beds
    return dict(
        product_slpm=prod_slpm,
        feed_air_slpm=feed_slpm,
        air_factor=feed_mol_s / prod_mol_s,
        pressure_work_w=w_iso,
        specific_wh_per_l=w_iso / (prod_slpm / 60.0) / 3600.0,  # W / (L/s) = J/L -> Wh/L
        cms_l_both_beds=v_cms_l,
        productivity_slpm_per_l_cms=prod_slpm / v_cms_l,
        p_header_barg=float(case.p_feed) - P_ATM,
        temperature_c=T - 273.15,
    )
