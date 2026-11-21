"""Phase 1 gates G1a, G1b, G1d, G1e, G1f for the sized plant (env viz). Writes gates_phase1.json.

Reads the final CSS state (model/css/final) and the 2N grid run (model/grid/2N); G1c is
model/regression/g1c.json. Run from anywhere: conda run -n viz python -I gates_phase1.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from psan2viz import css  # noqa: E402
from psan2viz.analysis import Cycle  # noqa: E402
from psan2viz.cases import P_DELIVER  # noqa: E402

res = {}
CSS_FINAL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "model" / "css" / "final"
GRID_2N = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "model" / "grid" / "2N"
OUT_JSON = HERE / ("gates_phase1.json" if len(sys.argv) == 1 else "gates_dryrun.json")
case, pl, x = css.load(CSS_FINAL)
x_end, recs = pl.run_cycle(x)
met = pl.cycle_metrics(recs)
feed = np.array(met["feed_mol"])
sp = list(case.species)

# ---- G1a: per segment and component, |in - out - d inventory| / that component's feed per cycle
def worst_balance(rr):
    sys_imb = np.array([np.abs(r["sys_imbalance"]) for r in rr]) / feed[None]
    bed_imb = np.array([np.abs(b) for r in rr for b in r["bed_imbalance"]]) / feed[None]
    tank_imb = np.array([np.abs(r["tank_imbalance"]) for r in rr]) / feed[None]
    return float(sys_imb.max()), float(bed_imb.max()), float(tank_imb.max()), sys_imb.max(0).tolist()

css_sys, css_bed, css_tank, per_comp = worst_balance(recs)
xs = pl.initial_state()  # start-up transient: three plain cycles from the initial state
start = []
for _ in range(3):
    xs, rr = pl.run_cycle(xs)
    start.append(worst_balance(rr)[:3])
res["G1a"] = dict(css_cycle=dict(system=css_sys, per_bed=css_bed, receiver=css_tank,
                                 system_per_component=dict(zip(sp, per_comp))),
                  startup_cycles_max=float(np.max(start)), limit=1e-4)
res["G1a"]["pass"] = max(css_sys, css_bed, css_tank, res["G1a"]["startup_cycles_max"]) < 1e-4

# ---- G1b: cycle-to-cycle change and cycle balance
rms, mx = pl.norm_change(x_end, x)
cyc_bal = (np.abs(feed - np.array(met["vent_mol"]) - np.array(met["product_mol"])) / feed)
res["G1b"] = dict(css_change_rms=rms, css_change_max=mx, limit=1e-6,
                  cycle_balance_per_component=dict(zip(sp, cyc_bal.tolist())), cycle_balance_limit=5e-3)
res["G1b"]["pass"] = rms < 1e-6 and float(cyc_bal.max()) < 5e-3

# ---- G1d: symmetry
cyc = Cycle(pl, recs)
sym = cyc.symmetry(240)
res["G1d"] = dict(max_scaled_diff=sym, limit=1e-4, pass_=sym < 1e-4)
res["G1d"]["pass"] = res["G1d"].pop("pass_")

# ---- G1e: grid (N vs 2N)
g2 = GRID_2N / "metrics.json"
if g2.exists():
    m2 = json.loads(g2.read_text())
    dO2 = abs(m2["product_O2_ppm"] / met["product_O2_ppm"] - 1)
    dR = 100 * abs(m2["recovery_N2"] - met["recovery_N2"])
    res["G1e"] = dict(N=[pl.N, 2 * pl.N], product_O2_ppm=[met["product_O2_ppm"], m2["product_O2_ppm"]],
                      recovery_N2=[met["recovery_N2"], m2["recovery_N2"]], d_O2_rel=dO2, d_recovery_pp=dR,
                      limits=dict(d_O2_rel=0.05, d_recovery_pp=1.0))
    res["G1e"]["pass"] = dO2 < 0.05 and dR < 1.0
else:
    res["G1e"] = dict(pass_=None, note="model/grid/2N not run yet")

# ---- G1f: sanity
ts, frames = cyc.sample(40)
wf = cyc.h2o_front(frames)
z_front, q_cms = wf["front_zL"], wf["cms_max_rel"]
eq = cyc.eq_residual()
p_tank_min = min(f["p_tank"] for f in frames)
res["G1f"] = dict(
    product_O2_ppm=met["product_O2_ppm"], O2_limit_ppm=5000.0,
    air_factor=met["air_factor"], air_factor_band=[2.0, 3.5],
    h2o_loading_front_zL=z_front, alumina_top_zL=cyc.z_al, h2o_on_cms_max_rel_to_feed_eq=q_cms,
    h2o_interface_y_rel_during_production=wf["interface_y_rel"],
    product_CO2_ppm=met["product_CO2_ppm"], product_H2O_ppm=met["product_H2O_ppm"],
    eq_residual=[e["ratio"] for e in eq], eq_limit=0.05,
    p_tank_min_bar=p_tank_min, p_deliver_bar=P_DELIVER,
    product_Ar_pct=met["product_Ar_pct"], N2_excl_Ar_pct=met["N2_excl_Ar_pct"], recovery_N2=met["recovery_N2"],
    productivity_Nm3_h_m3=met["productivity_Nm3_h_m3"])
f = res["G1f"]
f["checks"] = dict(O2=f["product_O2_ppm"] <= 5000.0, air_factor=2.0 <= f["air_factor"] <= 3.5,
                   h2o_in_alumina=q_cms < 0.05 and wf["interface_y_rel"] < 0.05,
                   co2_negligible=f["product_CO2_ppm"] < 0.01 * 1e6 * case.y_feed[sp.index("CO2")],
                   eq_nearly_equal=max(f["eq_residual"]) <= 0.05, regulator_supplied=p_tank_min > P_DELIVER + 0.2)
f["pass"] = all(f["checks"].values())
res["psa_sim_crosscheck"] = ("skipped: psa-sim supports several columns, equalization and >2 components, but not this case "
                            "(no simultaneous top+bottom equalization, no layered bed, no bidirectional product valve / "
                            "back-flow, no Darken LDF); per docs/MISSION.md")
g1c = json.loads((ROOT / "model" / "regression" / "g1c.json").read_text())
res["G1c"] = {"source": "model/regression/g1c.json", "max_diff_rtol_1e-9": g1c["rtol_1e-09"]["max_diff"], "pass_": g1c["G1c_pass"]}
res["G1c"]["pass"] = res["G1c"].pop("pass_")
res["all_pass"] = all(v.get("pass") is True for k, v in res.items() if isinstance(v, dict))
OUT_JSON.write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
