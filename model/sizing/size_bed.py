"""Size the beds for the user's spec (env viz): 30 SLPM at product O2 <= 0.5 %.

Fixed: 100 mm columns, the literature cycle (1 s + 59 s halves), feed/product/vent valves,
the receiver (6 bed volumes). Adjusted, one warm-started CSS solve per iteration:
  * h_cms   so that product O2 = O2_TARGET (10 % under the 5000 ppm spec; secant in ln O2)
  * cv_purge so that the purge per step = 40 % of the packed-bed volume at normal
            conditions (M21d reference process, Tab. 4.3-2)
  * h_al    grown by 30 % if the H2O front (y_H2O >= 1 % of the feed) passes 80 % of the layer
Writes runs/iterNN/ (CSS states) and sizing.json. Set OPENBLAS_NUM_THREADS.

usage: python size_bed.py [--start '{"h_cms": 0.6, ...}'] [--warm DIR] [--max-iter 8]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from psan2viz import css  # noqa: E402
from psan2viz.analysis import Cycle  # noqa: E402
from psan2viz.cases import n2_generator_case  # noqa: E402
from psan2viz.plant import Plant  # noqa: E402

O2_TARGET = 4500.0  # ppm
O2_TOL = 0.03
PURGE_X = 0.40  # bed volumes (normal conditions) per purge step, M21d
N_NORMAL = 101325.0 / (8.314462618 * 273.15)  # mol/m^3 at 0 degC, 1 atm
SOLVER = dict(rtol=1e-6, atol_rel=1e-9, atol_rel_trace=1e-6)

ap = argparse.ArgumentParser()
ap.add_argument("--start", default='{"h_cms": 0.60, "h_al": 0.06, "cv_purge": 2.5e-4}')
ap.add_argument("--warm", default=None)
ap.add_argument("--max-iter", type=int, default=8)
ap.add_argument("--m", type=int, default=20)
a = ap.parse_args()
logf = open(HERE / "sizing.log", "a")


def log(msg):
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()


params = json.loads(a.start)
x0 = np.load(Path(a.warm) / "css_state.npz")["x_start"] if a.warm else None
history = []
for it in range(1, a.max_iter + 1):
    case = n2_generator_case(**SOLVER, **params)
    pl = Plant(case)
    log(f"iter {it}: {json.dumps(params)}")
    t0 = time.time()
    x, hist = pl.run_to_css_proj(x0=x0 if (x0 is not None and x0.size == pl.size) else None, tol=2e-7,
                                 plain=0 if x0 is not None else 12,
                                 progress=lambda h: log(f"    {h['kind']} {h['iter']:3d} rms {h['rms']:.2e} maps {h['maps']} "
                                                        f"{h['wall_s']:.0f}s {h.get('note', '')}"))
    x_end, recs = pl.run_cycle(x)
    met = pl.cycle_metrics(recs)
    rms, mx = pl.norm_change(x_end, x)
    cyc = Cycle(pl, recs)
    _, frames = cyc.sample(20)
    z_front, q_cms = cyc.h2o_front(frames)
    purge_step = 0.5 * float(np.sum(met["purge_mol"]))
    purge_target = PURGE_X * pl.g.volume * N_NORMAL
    rec = dict(iter=it, params=dict(params), O2_ppm=met["product_O2_ppm"], air_factor=met["air_factor"],
               recovery_N2=met["recovery_N2"], purge_step_mol=purge_step, purge_target_mol=purge_target,
               h2o_front_zL=z_front, alumina_top_zL=cyc.z_al, h2o_cms_max_rel=q_cms, css_iterations=len(hist),
               css_change_rms=rms, wall_s=time.time() - t0, p_tank_min=None)
    history.append(rec)
    css.save(dict(case=case, x=x, hist=hist, metrics=met), HERE / "runs" / f"iter{it:02d}")
    log("  -> " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items() if k != "params"}))
    ok_o2 = abs(met["product_O2_ppm"] / O2_TARGET - 1) < O2_TOL
    ok_purge = abs(purge_step / purge_target - 1) < 0.03
    ok_h2o = z_front <= 0.8 * cyc.z_al
    if ok_o2 and ok_purge and ok_h2o:
        log("  sized.")
        break
    # updates
    params["cv_purge"] = params["cv_purge"] * purge_target / purge_step
    if not ok_h2o:
        params["h_al"] = round(params["h_al"] * 1.3, 4)
    pts = [(h["params"]["h_cms"], math.log(h["O2_ppm"])) for h in history]
    if len(pts) >= 2 and abs(pts[-1][0] - pts[-2][0]) > 1e-4:
        slope = (pts[-1][1] - pts[-2][1]) / (pts[-1][0] - pts[-2][0])
    else:
        slope = -6.0  # 1/m, first guess (ln O2 per metre of CMS)
    slope = min(slope, -1.0)
    dh = (math.log(O2_TARGET) - pts[-1][1]) / slope
    params["h_cms"] = round(float(np.clip(params["h_cms"] + dh, 0.5 * params["h_cms"], 1.6 * params["h_cms"])), 4)
    x0 = x
(HERE / "sizing.json").write_text(json.dumps(dict(target_O2_ppm=O2_TARGET, history=history, final=history[-1]), indent=1))
