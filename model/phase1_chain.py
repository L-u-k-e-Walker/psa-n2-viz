"""Phase 1 tail, unattended (env viz): 180-cell sizing refinement -> tight polish -> 360-cell grid check -> gates.

User decision (docs/MISSION.md, G1e): first-order upwind with 180 CMS + 30 alumina cells,
checked against 360 + 60. Steps (each logged to model/phase1_chain.log; resumable: a step whose
output exists is skipped):
  1. upsample the 60-cell sized state (model/sizing/sizing_runs_n60.json, its last iteration) to 180 cells
  2. size_bed.py at 180 cells (CSS tolerance 1e-5), warm-started, up to 3 iterations
  3. run_css.py -> model/css/final at tight tolerances (rtol 1e-8, atol 1e-11, trace 1e-8), CSS tol 2e-7 (G1b)
  4. grid/run_2n.py (360 + 60 cells, rtol 1e-6, CSS tol 1e-5) for G1e
  5. gates/gates_phase1.py
usage: OPENBLAS_NUM_THREADS=10 python phase1_chain.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = [sys.executable, "-I"]
LOG = HERE / "phase1_chain.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd, cwd):
    log("run: " + " ".join(str(c) for c in cmd))
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=cwd)
    log(f"  exit {r.returncode} after {time.time() - t0:.0f} s")
    if r.returncode != 0:
        sys.exit(r.returncode)


n60 = json.loads((HERE / "sizing" / "sizing_runs_n60.json").read_text())
p60 = n60["final"]["params"]
it60 = n60["final"]["iter"]
hist = n60["history"]
slope = -6.0
if len(hist) >= 2:
    import math
    a, b = hist[-2], hist[-1]
    if abs(b["params"]["h_cms"] - a["params"]["h_cms"]) > 1e-4:
        slope = (math.log(b["O2_ppm"]) - math.log(a["O2_ppm"])) / (b["params"]["h_cms"] - a["params"]["h_cms"])
log(f"60-cell sizing: {p60}, O2 {n60['final']['O2_ppm']:.0f} ppm, slope {slope:.2f} /m")
start = HERE / "sizing" / "n180_start.npy"
if not start.exists():
    run(PY + [HERE / "upsample.py", HERE / "sizing" / "runs_n60" / f"iter{it60:02d}", start,
              json.dumps({"n_al": 30, "n_cms": 180})], ROOT)
s180 = HERE / "sizing" / "sizing_runs_n180.json"
if not s180.exists():
    params = dict(p60, n_al=30, n_cms=180)
    run(PY + [HERE / "sizing" / "size_bed.py", "--start", json.dumps(params), "--warm", start, "--out", "runs_n180",
              "--max-iter", "3", "--slope", f"{min(slope, -1.0):.3f}"], HERE / "sizing")
n180 = json.loads(s180.read_text())
p180 = n180["final"]["params"]
log(f"180-cell sizing: {p180}, O2 {n180['final']['O2_ppm']:.0f} ppm")
final = HERE / "css" / "final" / "metrics.json"
if not final.exists():
    case = dict(p180, rtol=1e-8, atol_rel=1e-11, atol_rel_trace=1e-8)
    run(PY + [HERE / "css" / "run_css.py", HERE / "css" / "final", "--case", json.dumps(case), "--warm",
              HERE / "sizing" / "runs_n180" / f"iter{n180['final']['iter']:02d}", "--tol", "2e-7", "--plain", "0"], HERE / "css")
if not (HERE / "grid" / "2N" / "metrics.json").exists():
    run(PY + [HERE / "grid" / "run_2n.py", "--rtol", "1e-6", "--tol", "1e-5"], HERE / "grid")
run(PY + [HERE / "gates" / "gates_phase1.py"], HERE / "gates")
log("chain done")
