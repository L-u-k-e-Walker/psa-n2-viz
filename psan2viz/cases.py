"""The N2-generator case used for the renders, with the source of every input.

Nothing here is measured. Values are literature values (cited), fits to literature data
(model/fits/fits.json, made by model/fits/fit_inputs.py), calculated, or assumed with a
reason; docs/INPUTS.md lists them all. Component order: N2, O2, Ar, CO2, H2O.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .bed import BedGrid, Layer
from .isotherm import R_GAS, Sorbent
from .plant import PlantCase, Valves

ROOT = Path(__file__).resolve().parents[1]
FITS = ROOT / "model" / "fits" / "fits.json"
SPECIES = ("N2", "O2", "Ar", "CO2", "H2O")
T = 293.15  # K, isothermal (M21d reference temperature)
P_ATM = 1.01325  # bar
P_HEADER = 7.5 + P_ATM  # bar abs (user: 7.5 barg)
P_DELIVER = 6.0 + P_ATM  # bar abs, regulator outlet (assumed: ~1.5 bar below the inlet, typical of small generators)
STD_T, STD_P = 294.25, 101325.0  # user: SLPM at 21.1 degC, 1 atm
PRODUCT_SLPM = 30.0
PRODUCT_MOL_S = PRODUCT_SLPM / 60.0 / (R_GAS * STD_T / STD_P * 1000.0)  # 0.02071 mol/s
DEW_POINT_C = 3.0  # user: refrigerated dryer, ISO 8573-1 class 4


def psat_water(Tk):
    """IAPWS-IF97 saturation pressure (Pa)."""
    n = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2, 0.12020824702470e5, -0.32325550322333e7,
         0.14915108613530e2, -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849, 0.65017534844798e3]
    th = Tk + n[8] / (Tk - n[9])
    A, B, C = th * th + n[0] * th + n[1], n[2] * th * th + n[3] * th + n[4], n[5] * th * th + n[6] * th + n[7]
    return (2 * C / (-B + math.sqrt(B * B - 4 * A * C))) ** 4 * 1e6


def feed_composition(p_bar=P_HEADER, dew_c=DEW_POINT_C):
    """Wet feed after the dryer: y_H2O = f p_sat(T_dp) / P (IF97, Buck 1981 enhancement factor)."""
    P = p_bar * 1e5
    ps = psat_water(273.15 + dew_c)
    f = 1 + 1e-4 * (7.2 + P / 100 * (0.0320 + 5.9e-6 * dew_c**2))
    y_w = f * ps / P
    dry = dict(O2=0.20946, Ar=0.00934, CO2=0.00042)
    dry["N2"] = 1.0 - sum(dry.values())
    y = np.array([dry[s] * (1 - y_w) for s in SPECIES[:4]] + [y_w])
    return y / y.sum()


def load_solver_params(fits_path=FITS):
    return json.loads(Path(fits_path).read_text())["solver_params"]


def sorbents(params):
    d = params
    cms = Sorbent("CMS (Shirasagi MSC CT-350)", d["cms"]["qs"], d["cms"]["b"], d["cms"]["k0"], darken=True,
                  provenance={"isotherm": "dual-site Langmuir fits (model/fits)", "kinetics": "LDF + Darken (model/fits)"})
    alu = Sorbent("activated alumina (F-200-like)", d["alumina"]["qs"], d["alumina"]["b"], d["alumina"]["k0"],
                  darken=False, provenance={"isotherm": "fits to ISODB data (model/fits)", "kinetics": "Glueckauf"})
    return cms, alu


# layer data
CMS_BULK = 711.0  # kg/m^3, M21d Tab. 3.1-1
CMS_VOID = 0.543  # total bed voidage (inter-particle 0.404 + pellet pores), M21d Tab. 3.1-1; pore gas taken in equilibrium with the bulk gas
ALU_VOID_INTER, ALU_EPS_P, ALU_RHO_P = 0.40, 0.55, 1250.0  # assumed (F-200-like beads)
ALU_BULK = (1 - ALU_VOID_INTER) * ALU_RHO_P  # 750 kg/m^3 (F-200 data sheets: ~770)
ALU_VOID = ALU_VOID_INTER + (1 - ALU_VOID_INTER) * ALU_EPS_P  # 0.73 total voidage


def bed_grid(h_cms, h_al, radius, n_cms, n_al, params, scheme="upwind"):
    cms, alu = sorbents(params)
    layers = [Layer("alumina", alu, h_al, ALU_VOID, ALU_BULK, n_al), Layer("CMS", cms, h_cms, CMS_VOID, CMS_BULK, n_cms)]
    y = feed_composition()
    c_ref = P_HEADER * 1e5 / (R_GAS * T) * np.maximum(y, 1e-4)
    return BedGrid(layers, radius, T, scheme=scheme, va_eps=1e-6 * c_ref)


def eq_cv_for(grid, t_eq, residual=0.02):
    """Equalization Cv (each of the two connections) for |dP| to fall to ``residual`` of its start in t_eq.

    Capacity per bar of one bed = gas + fast O2 adsorption around the mid pressure; two connections
    and two beds: tau = cap / (4 Cv)."""
    V = grid.volume
    cap_gas = sum(lay.void * lay.height for lay in grid.layers) / grid.length * V * 1e5 / grid.RT
    return cap_gas * 1.5 * math.log(1 / residual) / (4 * t_eq)  # x1.5: O2 uptake adds ~50 % (checked by G1f)


def n2_generator_case(h_cms=0.60, h_al=0.06, radius=0.05, n_cms=60, n_al=10, t_eq=1.0, t_prod=59.0,
                      cv_feed=0.12, cv_product=0.06, cv_vent=0.15, cv_purge=2.5e-4, cv_eq=None,
                      tank_volume=None, product_mol_s=PRODUCT_MOL_S, solver_params=None, fits_path=FITS,
                      scheme="upwind", **kw) -> PlantCase:
    """Build the case. ``case_args`` on the result records every argument (and the sorbent
    parameters) so that ``case_from_args`` rebuilds it exactly, e.g. from a cached CSS state."""
    params = load_solver_params(fits_path) if solver_params is None else solver_params
    grid = bed_grid(h_cms, h_al, radius, n_cms, n_al, params, scheme)
    cv_eq = eq_cv_for(grid, t_eq) if cv_eq is None else cv_eq
    tank_volume = 6.0 * grid.volume if tank_volume is None else tank_volume  # M21d: 12 L receiver per 2 L bed
    prov = {
        "temperature": "assumed: 293.15 K isothermal (M21d reference temperature; user: isothermal)",
        "feed": f"user: header 7.5 barg, dryer pressure dew point +3 degC -> y_H2O {feed_composition()[4] * 1e6:.0f} ppm "
                "(IF97 + Buck 1981); dry air O2 20.946 %, Ar 0.934 %, CO2 420 ppm, N2 balance",
        "product": "user: 30 SLPM at 21.1 degC, 1 atm = 0.02071 mol/s through the outlet regulator",
        "p_deliver": "assumed: regulator outlet 6.0 barg (about 1.5 bar below the inlet, typical of small generators)",
        "bed radius": "assumed: 100 mm ID columns, typical of small CMS generators",
        "h_cms": "sized (model/sizing) so that product O2 meets 0.5 % at 30 SLPM",
        "h_al": "sized (model/sizing) so that the alumina captures >= 95 % of the water (CMS water < 5 % of its feed "
                "equilibrium, interface y_H2O < 5 % of the feed during production)",
        "CMS layer": "M21d Tab. 3.1-1: bulk density 711 kg/m^3, total voidage 0.543",
        "alumina layer": "assumed: F-200-like beads, inter-particle void 0.40, pellet porosity 0.55, pellet density 1250 kg/m^3",
        "cycle": "M21d Tab. 4.3-2: half-cycle 60 s = 1 s equalization (top/top + bottom/bottom) + 59 s production / "
                 "blowdown with simultaneous purge; product back-flow during pressurization",
        "valves": "assumed (sized): feed 0.12 and product 0.06 mol/(s bar) so the beds peak near 8 bar abs (M21d's "
                  "reference pressure); vent 0.15 (blowdown in a few seconds); equalization Cv for <= 2 % residual "
                  "pressure difference after 1 s; purge Cv calibrated to M21d's purge of 40 % of the bed volume per step",
        "receiver": "assumed: 6 x one bed volume (M21d's pilot: 12 L receiver for 2 L beds)",
        "grid": f"{n_al} alumina + {n_cms} CMS cells, first-order upwind (grid-checked, G1e)",
    }
    args = dict(grid=grid, species=SPECIES, y_feed=feed_composition(), p_feed=P_HEADER, p_vent=P_ATM,
                valves=Valves(feed=cv_feed, vent=cv_vent, product=cv_product, purge=cv_purge, eq_top=cv_eq, eq_bot=cv_eq),
                t_eq=t_eq, t_prod=t_prod, tank_volume=tank_volume, product_flow=product_mol_s, p_deliver=P_DELIVER,
                y_vent_in=None, provenance=prov)
    args.update(kw)
    case = PlantCase(**args)
    case.case_args = dict(h_cms=h_cms, h_al=h_al, radius=radius, n_cms=n_cms, n_al=n_al, t_eq=t_eq, t_prod=t_prod,
                          cv_feed=cv_feed, cv_product=cv_product, cv_vent=cv_vent, cv_purge=cv_purge, cv_eq=cv_eq,
                          tank_volume=tank_volume, product_mol_s=product_mol_s, solver_params=params, scheme=scheme,
                          **{k: v for k, v in kw.items() if isinstance(v, (int, float, str))})
    return case


def case_from_args(args):
    return n2_generator_case(**args)
