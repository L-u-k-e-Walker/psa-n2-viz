"""CSS with converged trace profiles (env viz): full-model cycles + frozen-bulk trace marches.

Loads a CSS state folder (case + state), optionally a warm .npy state, tightens the integrator
tolerances, and runs psan2viz.trace.converge: ``--full`` full-model cycles (the bulk follows the trace
state), then a frozen-bulk march of the CO2/H2O profiles on the last recorded cycle (up to
``--march-max`` cycles, ~1 s each at 60 cells), repeated until one full-model cycle changes the
normalized state by < tol (RMS). Verified by a final full-model cycle; writes OUT/ like run_css.py.
Set OPENBLAS_NUM_THREADS.

usage: python trace_css.py SRC OUT [--warm X.npy] [--tol 1e-6] [--rtol 1e-7] [--march-max 3000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from psan2viz import css  # noqa: E402
from psan2viz.cases import case_from_args  # noqa: E402
from psan2viz.plant import Plant  # noqa: E402
from psan2viz.trace import converge  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--warm", default=None)
ap.add_argument("--tol", type=float, default=1e-6)
ap.add_argument("--rtol", type=float, default=1e-7)
ap.add_argument("--march-max", type=int, default=3000)
ap.add_argument("--march-tol", type=float, default=None)
ap.add_argument("--full", type=int, default=3)
ap.add_argument("--outer-max", type=int, default=12)
ap.add_argument("--accel", action="store_true", help="single-mode jumps inside the trace marches (TraceMarch.march_accel)")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
logf = open(out / "run.log", "a")


def log(msg):
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()


d = np.load(Path(a.src) / "css_state.npz", allow_pickle=False)
args = json.loads(str(d["case_args"]))
args.update(rtol=a.rtol, atol_rel=a.rtol * 1e-3, atol_rel_trace=a.rtol)
case = case_from_args(args)
pl = Plant(case)
x0 = np.load(a.warm) if a.warm else d["x_start"]
assert x0.size == pl.size
log(f"src {a.src}; warm {a.warm}; tol {a.tol}; rtol {a.rtol}; march_max {a.march_max}; full {a.full}")
t0 = time.time()


def prog(h):
    if h["kind"] == "trace":
        log(f"    trace cycle {h['cycle']:5d}  rms {h['rms']:.3e}  max {h['max']:.3e}  {h['wall_s']:.0f}s  ({h['note']})")
    else:
        log(f"  {h['kind']:5s} {h['iter']:3d}  rms {h['rms']:.3e}  {h['wall_s']:.0f}s  {h.get('note', '')}")


x, hist = converge(pl, x0, tol=a.tol, march_tol=a.march_tol, march_max=a.march_max, full_between=a.full,
                   outer_max=a.outer_max, accel=a.accel, progress=prog, checkpoint=lambda xc: np.save(out / "checkpoint.npy", xc))
t_css = time.time() - t0
x_end, recs = pl.run_cycle(x)
rms, mx = pl.norm_change(x_end, x)
met = pl.cycle_metrics(recs)
met.update(css_change_rms=rms, css_change_max=mx, css_iterations=len(hist), wall_css_s=t_css, converged=bool(rms < a.tol))
css.save(dict(case=case, x=x, hist=hist, metrics=met), out)
sp = list(case.species)
bal = {s: (met["feed_mol"][i] - met["vent_mol"][i] - met["product_mol"][i]) / met["feed_mol"][i] for i, s in enumerate(sp)}
log("cycle balance (feed - vent - product) / feed: " + json.dumps({k: f"{v:.2e}" for k, v in bal.items()}))
log(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in met.items() if not k.endswith("_mol")}, indent=1))
