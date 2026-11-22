"""Design and performance table of a CSS state, as Markdown (env viz). For docs/INPUTS.md and the README.

Every row carries its source class from the case's provenance (user / literature / sized / assumed).
usage: conda run -n viz python -I tools/design_table.py [CSS_DIR]   (default model/css/final)
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from psan2viz import css  # noqa: E402
from psan2viz.performance import performance  # noqa: E402

src = (Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "model" / "css" / "final").resolve()
case, pl, x = css.load(src)
a = case.case_args
g = pl.g
met = json.loads((src / "metrics.json").read_text())
prov = case.provenance
area = math.pi * a["radius"] ** 2
mass = [L.bulk_density * L.height * area for L in g.layers]  # cell 0 = feed end: alumina first, then CMS
m_al, m_cms = mass[0], mass[-1]
v = case.valves
rows = [
    ("Column inner diameter", f"{2 * a['radius'] * 1e3:.0f} mm", prov["bed radius"]),
    ("CMS layer height", f"{a['h_cms']:.3f} m ({m_cms:.2f} kg per bed)", prov["h_cms"]),
    ("Alumina layer height (feed end)", f"{a['h_al'] * 100:.1f} cm ({m_al:.2f} kg per bed)", prov["h_al"]),
    ("Packed volume per bed", f"{g.volume * 1e3:.2f} L", "from the above"),
    ("Cycle", f"{2 * (a['t_eq'] + a['t_prod']):.0f} s: {a['t_eq']:.0f} s equalization + {a['t_prod']:.0f} s production per half",
     prov["cycle"]),
    ("Feed valve", f"{v.feed:.3g} mol/(s bar)", "assumed: sized so the beds peak near 8 bar abs (M21d reference pressure)"),
    ("Product valve", f"{v.product:.3g} mol/(s bar)", "assumed: as the feed valve"),
    ("Vent valve", f"{v.vent:.3g} mol/(s bar)", "assumed: blowdown in a few seconds"),
    ("Purge valve", f"{v.purge:.3g} mol/(s bar)", "sized: purge of 40 % of the bed volume per step (M21d Tab. 4.3-2)"),
    ("Equalization valves (top, bottom)", f"{v.eq_top:.3g} mol/(s bar) each", "sized: <= 2 % residual pressure difference after 1 s"),
    ("Receiver", f"{case.tank_volume * 1e3:.1f} L", prov["receiver"]),
    ("Delivery pressure", f"{case.p_deliver - 1.01325:.2f} barg", prov["p_deliver"]),
    ("Grid", f"{a['n_al']} alumina + {a['n_cms']} CMS cells", prov["grid"]),
]
pf = performance(met, case)
perf = [
    ("Product rate", f"{pf['product_slpm']:.1f} SLPM N2 (21.1 degC, 1 atm)"),
    ("Product O2", f"{met['product_O2_ppm']:,.0f} ppm (spec <= 5,000)"),
    ("Purity, N2 + Ar (industry)", f"{met['purity_N2_plus_Ar_pct']:.2f} %"),
    ("N2 excluding Ar", f"{met['N2_excl_Ar_pct']:.2f} % (Ar {met['product_Ar_pct']:.2f} %)"),
    ("Product CO2, H2O", f"{met['product_CO2_ppm']:.2g} ppm, {met['product_H2O_ppm']:.2g} ppm"),
    ("N2 recovery", f"{100 * met['recovery_N2']:.1f} %"),
    ("Feed air", f"{pf['feed_air_slpm']:.0f} SLPM at {pf['p_header_barg']:.1f} barg (air factor {pf['air_factor']:.2f})"),
    ("Pressure work (ideal isothermal compression of the feed air)", f"{pf['pressure_work_w']:.0f} W = {pf['specific_wh_per_l']:.3f} Wh per L N2"),
    ("Productivity", f"{pf['productivity_slpm_per_l_cms']:.2f} SLPM per L of CMS ({pf['cms_l_both_beds']:.1f} L CMS in both beds)"),
    ("Purge / product", f"{met['purge_to_product']:.3f}"),
    ("CSS change of one full cycle", f"{met['css_change_rms']:.1e} rms, {met['css_change_max']:.1e} max (normalized)"),
]
print(f"Design ({src.relative_to(ROOT)})\n")
print("| Quantity | Value | Source |\n|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]} |")
print("\nPerformance at the cyclic steady state (computed)\n")
print("| Quantity | Value |\n|---|---|")
for r in perf:
    print(f"| {r[0]} | {r[1]} |")
