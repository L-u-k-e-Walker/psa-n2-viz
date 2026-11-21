"""Phase 1 tail, unattended (env viz): 180-cell sizing -> 360-cell grid check (G1e).

User decision (docs/MISSION.md, G1e): first-order upwind with 180 CMS + 30 alumina cells,
checked against 360 + 60. Steps (logged to model/phase1_chain.log; a step whose output exists is
skipped, so the chain can be restarted):
  0. wait for the 60-cell sizing (model/sizing/sizing_runs_n60c.json)
  1. upsample its last iteration to 180 cells
  2. size_bed.py at 180 cells (CSS tolerance 5e-5: product O2 and recovery are converged far below
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
sys.path.insert(0, str(HERE / "sizing"))
from sizing_rule import next_params, sized  # noqa: E402
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


j60 = HERE / "sizing" / "sizing_runs_n60c.json"
log("waiting for the 60-cell sizing" if not j60.exists() else "60-cell sizing found")
while not j60.exists():
    time.sleep(30)
n60 = json.loads(j60.read_text())
p60 = n60["final"]["params"]
it60 = n60["final"]["iter"]
hist = n60["history"]
slope = -3.52  # measured between 0.60 and 0.93 m of CMS at 60 cells (docs/LOG.md)
good = [h for h in hist if abs(h["purge_step_mol"] / h["purge_target_mol"] - 1) < 0.05
        and abs(h["params"]["h_al"] - p60["h_al"]) < 1e-6]  # same filter as size_bed.py
if len(good) >= 2:
    a, b = good[-2], good[-1]
    if abs(b["params"]["h_cms"] - a["params"]["h_cms"]) > 1e-4:
        slope = (math.log(b["O2_ppm"]) - math.log(a["O2_ppm"])) / (b["params"]["h_cms"] - a["params"]["h_cms"])
log(f"60-cell sizing: {p60}, O2 {n60['final']['O2_ppm']:.0f} ppm, slope {slope:.2f} /m")
start = HERE / "sizing" / "n180_start.npy"
if not start.exists():
    run(PY + [HERE / "upsample.py", HERE / "sizing" / "runs_n60c" / f"iter{it60:02d}", start,
              json.dumps({"n_al": 30, "n_cms": 180})], ROOT)
def size_running(out):
    r = subprocess.run(["pgrep", "-f", f"size_bed.py.*--out {out}"], capture_output=True, text=True)
    return bool(r.stdout.strip())


# 180-cell sizing: rounds runs_n180, runs_n180b, ... until sized (a round already running is waited for)
rnd, prev, rounds, done_outs = 0, None, [], []
SLOPE_180 = -6.47  # d ln(O2)/d h_cms at 180 cells, from 1.064 m (1,284 ppm) and 0.705 m (13,177 ppm)
TRACE_SRC = HERE / "css" / "runs" / "trace60"  # converged trace profiles (60 cells)
while True:
    out = "runs_n180" + ("" if rnd == 0 else chr(ord("a") + rnd - 1))
    js = HERE / "sizing" / f"sizing_{out}.json"
    if not js.exists():
        if size_running(out):
            log(f"waiting for the running sizing round {out}")
            while size_running(out) and not js.exists():
                time.sleep(30)
        else:
            priors = [HERE / "sizing" / f"sizing_{o}.json" for o in done_outs]
            if prev is None:
                params, warm, sl = dict(p60, n_al=30, n_cms=180), start, slope
            else:
                # continue with the shared update rule over every 180-cell point so far
                params, sl = next_params([r for rr in rounds for r in rr["history"]], SLOPE_180)
                warm = HERE / "sizing" / prev_out / f"iter{prev['final']['iter']:02d}"
            run(PY + [HERE / "sizing" / "size_bed.py", "--start", json.dumps(params), "--warm", warm, "--out", out,
                      "--max-iter", "4", "--tol", "5e-5", "--max-maps", "40", "--trace-src", TRACE_SRC,
                      "--slope", f"{min(sl, -1.0):.3f}", "--prior", *priors], HERE / "sizing")
    if not js.exists():
        log(f"sizing round {out} ended without its json")
        sys.exit(1)
    cur = json.loads(js.read_text())
    log(f"sizing round {out}: final {cur['final']['params']}, O2 {cur['final']['O2_ppm']:.0f} ppm, sized {sized(cur['final'])}")
    if sized(cur["final"]):
        break
    rounds.append(cur)
    done_outs.append(out)
    prev, prev_out, rnd = cur, out, rnd + 1
    if rnd > 3:
        log("180-cell sizing not converged after 4 rounds")
        sys.exit(1)
n180 = cur
src180 = HERE / "sizing" / out / f"iter{n180['final']['iter']:02d}"
log(f"180-cell sizing: {n180['final']['params']}, O2 {n180['final']['O2_ppm']:.0f} ppm, state {src180}")
if not (HERE / "grid" / "2N" / "metrics.json").exists():
    run(PY + [HERE / "grid" / "run_2n.py", "--src", src180, "--rtol", "1e-6", "--tol", "5e-5", "--max-maps", "30", "--trace-src", TRACE_SRC], HERE / "grid")
log("chain done (next: final 180-cell state with the trace solver, then the gates)")
