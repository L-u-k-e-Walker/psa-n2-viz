"""Gate G1e: the sized case at twice the cells (env viz).

Reads model/css/final (the N-cell CSS), rebuilds the same case with 2N cells in each layer,
warm-starts from the N solution with every cell duplicated (piecewise-constant refinement), and
solves the CSS with the same method and tolerances. Writes model/grid/2N/ (css_state.npz,
metrics.json, css_history.csv, run.log). Set OPENBLAS_NUM_THREADS.
"""
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
from psan2viz.trace import transfer  # noqa: E402

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rtol", type=float, default=1e-6, help="integrator tolerance for the 2N run")
ap.add_argument("--tol", type=float, default=1e-5, help="CSS tolerance (product O2 and recovery settle far earlier)")
ap.add_argument("--src", default=str(ROOT / "model" / "css" / "final"), help="N-cell CSS state to refine")
ap.add_argument("--max-maps", type=int, default=3000, help="cap on half-cycle maps (the slow trace mode does not\n                move product O2 or recovery, the G1e quantities)")
ap.add_argument("--trace-src", default=None, help="CSS state with converged trace profiles to map onto the 2N grid")
a = ap.parse_args()
out = HERE / "2N"
out.mkdir(parents=True, exist_ok=True)
logf = open(out / "run.log", "a")


def log(msg):
    print(msg, flush=True)
    logf.write(msg + "\n")
    logf.flush()


d = np.load(Path(a.src) / "css_state.npz", allow_pickle=False)
args = json.loads(str(d["case_args"]))
case1, pl1, x1 = css.load(Path(a.src))
args2 = dict(args, n_al=2 * args["n_al"], n_cms=2 * args["n_cms"], rtol=a.rtol, atol_rel=1e-9, atol_rel_trace=1e-6)
case2 = case_from_args(args2)
pl2 = Plant(case2)
x2 = np.zeros(pl2.size)
for k in (0, 1):
    c1, q1 = pl1.bed(x1, k)
    c2, q2 = pl2.bed(x2, k)
    c2[:] = np.repeat(c1, 2, axis=1)
    q2[:] = np.repeat(q1, 2, axis=1)
x2[pl2.s_t] = x1[pl1.s_t]
if a.trace_src:
    _, plT, xT = css.load(Path(a.trace_src))
    x2 = transfer(plT, xT, pl2, x2)
log(f"2N run: n_al {args2['n_al']}, n_cms {args2['n_cms']}, state {pl2.size}")
t0 = time.time()
x, hist = pl2.run_to_css_proj(x0=x2, tol=a.tol, plain=0, cos_min=0.9, m_max=10.0, max_maps=a.max_maps, checkpoint=lambda xc: np.save(out / "checkpoint.npy", xc),
                              progress=lambda h: log(f"  {h['kind']} {h['iter']:3d} rms {h['rms']:.3e} maps {h['maps']} "
                                                     f"{h['wall_s']:.0f}s {h.get('note', '')}"))
x_end, recs = pl2.run_cycle(x)
rms, mx = pl2.norm_change(x_end, x)
met = pl2.cycle_metrics(recs)
met.update(css_change_rms=rms, css_change_max=mx, css_iterations=len(hist), wall_css_s=time.time() - t0)
css.save(dict(case=case2, x=x, hist=hist, metrics=met), out)
log(json.dumps({k: v for k, v in met.items() if not k.endswith("_mol")}, indent=1, default=float))
