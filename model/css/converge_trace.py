"""Final CSS convergence of a sized state (env viz): deflate the slow trace-profile mode.

Loads a CSS state folder (case + state, e.g. model/sizing/runs_n180/iterNN), optionally tightens the
integrator tolerances, and runs Plant.run_to_css_mode (plain full cycles + single-mode jumps along the
slow eigenvector: the water front in the alumina with CO2 displaced along it). Writes OUT/ with
css_state.npz, metrics.json, css_history.csv and run.log, like run_css.py. Set OPENBLAS_NUM_THREADS.

usage: python converge_trace.py SRC OUT [--tol 1e-6] [--rtol 1e-7] [--relax 6] [--max-step 0.05]
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

ap = argparse.ArgumentParser()
ap.add_argument("src", help="CSS state folder to start from")
ap.add_argument("out")
ap.add_argument("--warm", default=None, help="optional .npy state to start from instead of SRC's state")
ap.add_argument("--tol", type=float, default=1e-6, help="RMS of the normalized full-cycle change (G1b)")
ap.add_argument("--rtol", type=float, default=None, help="integrator rtol (atol_rel = rtol/1e3, trace rtol)")
ap.add_argument("--relax", type=int, default=6)
ap.add_argument("--cos-min", type=float, default=0.995)
ap.add_argument("--max-step", type=float, default=0.05)
ap.add_argument("--max-cycles", type=int, default=2000)
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
if a.rtol:
    args.update(rtol=a.rtol, atol_rel=a.rtol * 1e-3, atol_rel_trace=a.rtol)
case = case_from_args(args)
pl = Plant(case)
x0 = np.load(a.warm) if a.warm else d["x_start"]
assert x0.size == pl.size, (x0.size, pl.size)
log(f"src {a.src}; case {json.dumps({k: v for k, v in args.items() if k != 'solver_params'})}")
log(f"tol {a.tol}, relax {a.relax}, cos_min {a.cos_min}, max_step {a.max_step}")
t0 = time.time()
x, hist = pl.run_to_css_mode(x0, tol=a.tol, relax=a.relax, cos_min=a.cos_min, max_step=a.max_step,
                             max_cycles=a.max_cycles, checkpoint=lambda xc: np.save(out / "checkpoint.npy", xc),
                             progress=lambda h: log(f"  {h['kind']:5s} {h['iter']:3d}  rms {h['rms']:.3e}  max {h['max']:.3e}  "
                                                    f"cycles {h['cycles']:4d}  {h['wall_s']:.0f}s  {h.get('note', '')}"))
t_css = time.time() - t0
x_end, recs = pl.run_cycle(x)
rms, mx = pl.norm_change(x_end, x)
met = pl.cycle_metrics(recs)
met.update(css_change_rms=rms, css_change_max=mx, css_iterations=len(hist), wall_css_s=t_css,
           converged=bool(rms < a.tol))
css.save(dict(case=case, x=x, hist=hist, metrics=met), out)
log(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in met.items() if not k.endswith("_mol")}, indent=1))
