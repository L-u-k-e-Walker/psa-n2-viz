"""Solve the plant's cyclic steady state and save it (env viz).

usage: python run_css.py OUT_DIR [--case '{"h_cms": 0.6, ...}'] [--warm DIR] [--m 20] [--tol 2e-7]
Writes OUT_DIR/css_state.npz (state at the start of the cycle + every case argument),
metrics.json (performance of the verification cycle, CSS diagnostics), css_history.csv, run.log.
Set OPENBLAS_NUM_THREADS (8-10 for one run, 1 per worker for parallel runs).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from psan2viz import css  # noqa: E402
from psan2viz.cases import n2_generator_case  # noqa: E402
from psan2viz.plant import Plant  # noqa: E402

DEFAULTS = dict(rtol=1e-6, atol_rel=1e-9, atol_rel_trace=1e-6)

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--case", default="{}")
ap.add_argument("--warm", default=None, help="folder with a css_state.npz of the same cell counts")
ap.add_argument("--m", type=int, default=20)
ap.add_argument("--plain", type=int, default=12)
ap.add_argument("--tol", type=float, default=2e-7)
ap.add_argument("--max-iter", type=int, default=300)
ap.add_argument("--method", default="proj", choices=["proj", "nk", "anderson"])
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
logf = open(out / "run.log", "a")


def log(msg):
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()


kw = dict(DEFAULTS, **json.loads(a.case))
case = n2_generator_case(**kw)
pl = Plant(case)
x0 = None
if a.warm:
    w = Path(a.warm)
    x0 = np.load(w) if w.suffix == ".npy" else np.load(w / "css_state.npz")["x_start"]
    if x0.size != pl.size:
        log(f"warm start ignored: size {x0.size} != {pl.size}")
        x0 = None
log(f"case {json.dumps({k: v for k, v in case.case_args.items() if k != 'solver_params'})}")
log(f"state size {pl.size}, Anderson m={a.m}, plain={a.plain}, tol={a.tol}, warm={a.warm}")
t0 = time.time()
def checkpoint(xc):
    np.save(out / "checkpoint.npy", xc)


if a.method == "proj":
    x, hist = pl.run_to_css_proj(x0=x0, tol=a.tol, plain=a.plain, checkpoint=checkpoint, progress=lambda h: log(
        f"  {h['kind']:6s} {h['iter']:3d}  rms {h['rms']:.3e}  max {h['max']:.3e}  maps {h['maps']:4d}  "
        f"{h['wall_s']:.0f}s  {h.get('note', '')}"))
elif a.method == "nk":
    x, hist = pl.run_to_css_nk(x0=x0, tol=a.tol, plain=a.plain, progress=lambda h: log(
        f"  {h['kind']:6s} {h['iter']:3d}  rms {h['rms']:.3e}  max {h['max']:.3e}  maps {h['maps']:4d}  "
        f"{h['wall_s']:.0f}s  {h.get('note', '')}"))
else:
    x, hist = pl.run_to_css(x0=x0, tol=a.tol, max_iter=a.max_iter, m=a.m, plain=a.plain,
                            progress=lambda h: log(f"  iter {h['iter']:3d}  rms {h['rms']:.3e}  max {h['max']:.3e}  {h['wall_s']:.0f}s"))
t_css = time.time() - t0
x_end, recs = pl.run_cycle(x)
rms, mx = pl.norm_change(x_end, x)
met = pl.cycle_metrics(recs)
met.update(css_change_rms=rms, css_change_max=mx, css_iterations=len(hist), wall_css_s=t_css,
           wall_verify_cycle_s=time.time() - t0 - t_css, converged=bool(hist[-1]["rms"] < a.tol))
css.save(dict(case=case, x=x, hist=hist, metrics=met), out)
log(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in met.items()
                if not k.endswith("_mol")}, indent=1))
