"""Phase 1 tail, unattended (env viz): 180-cell sizing -> 360-cell grid check (G1e).

User decision (docs/MISSION.md, G1e): first-order upwind with 180 CMS + 30 alumina cells,
checked against 360 + 60. Steps (logged to model/phase1_chain.log; a step whose output exists is
skipped, so the chain can be restarted):
  0. wait for the 60-cell sizing (model/sizing/sizing_runs_n60b.json)
  1. upsample its last iteration to 180 cells
  2. size_bed.py at 180 cells (CSS tolerance 3e-5: product O2 and recovery are converged far below
     that; only the slow trace profiles remain), warm-started, up to 3 iterations
  3. grid/run_2n.py from the 180-cell sizing state (360 + 60 cells) for G1e
The final 180-cell state with converged trace profiles (G1b) is made afterwards with the trace
solver; the gates run last.
usage: OPENBLAS_NUM_THREADS=10 python phase1_chain.py
"""
import json
import math
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


j60 = HERE / "sizing" / "sizing_runs_n60b.json"
log("waiting for the 60-cell sizing" if not j60.exists() else "60-cell sizing found")
while not j60.exists():
    time.sleep(30)
n60 = json.loads(j60.read_text())
p60 = n60["final"]["params"]
it60 = n60["final"]["iter"]
hist = n60["history"]
slope = -3.52  # measured between 0.60 and 0.93 m of CMS at 60 cells (docs/LOG.md)
if len(hist) >= 2:
    a, b = hist[-2], hist[-1]
    if abs(b["params"]["h_cms"] - a["params"]["h_cms"]) > 1e-4:
        slope = (math.log(b["O2_ppm"]) - math.log(a["O2_ppm"])) / (b["params"]["h_cms"] - a["params"]["h_cms"])
log(f"60-cell sizing: {p60}, O2 {n60['final']['O2_ppm']:.0f} ppm, slope {slope:.2f} /m")
start = HERE / "sizing" / "n180_start.npy"
if not start.exists():
    run(PY + [HERE / "upsample.py", HERE / "sizing" / "runs_n60b" / f"iter{it60:02d}", start,
              json.dumps({"n_al": 30, "n_cms": 180})], ROOT)
s180 = HERE / "sizing" / "sizing_runs_n180.json"
if not s180.exists():
    params = dict(p60, n_al=30, n_cms=180)
    run(PY + [HERE / "sizing" / "size_bed.py", "--start", json.dumps(params), "--warm", start, "--out", "runs_n180",
              "--max-iter", "3", "--tol", "3e-5", "--slope", f"{min(slope, -1.0):.3f}"], HERE / "sizing")
n180 = json.loads(s180.read_text())
src180 = HERE / "sizing" / "runs_n180" / f"iter{n180['final']['iter']:02d}"
log(f"180-cell sizing: {n180['final']['params']}, O2 {n180['final']['O2_ppm']:.0f} ppm, state {src180}")
if not (HERE / "grid" / "2N" / "metrics.json").exists():
    run(PY + [HERE / "grid" / "run_2n.py", "--src", src180, "--rtol", "1e-6", "--tol", "3e-5"], HERE / "grid")
log("chain done (next: final 180-cell state with the trace solver, then the gates)")
