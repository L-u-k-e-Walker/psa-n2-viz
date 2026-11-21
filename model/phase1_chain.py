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
def sized(rec):
    """Same acceptance as size_bed.py."""
    return (abs(rec["O2_ppm"] / 4500.0 - 1) < 0.03 and abs(rec["purge_step_mol"] / rec["purge_target_mol"] - 1) < 0.03
            and rec["h2o_cms_max_rel"] < 0.05 and rec["h2o_interface_y_rel"] < 0.05)


def size_running(out):
    r = subprocess.run(["pgrep", "-f", f"size_bed.py.*--out {out}"], capture_output=True, text=True)
    return bool(r.stdout.strip())


# 180-cell sizing: rounds runs_n180, runs_n180b, ... until sized (a round already running is waited for)
rnd, prev, rounds = 0, None, []
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
            if prev is None:
                params, warm, sl = dict(p60, n_al=30, n_cms=180), start, slope
            else:
                # continue from the last evaluated point with size_bed.py's update rule (secant in
                # ln O2 over the 180-cell points with the purge on target, purge valve rescaled)
                h = [r for rr in rounds for r in rr["history"]]
                f = prev["final"]
                warm = HERE / "sizing" / prev_out / f"iter{f['iter']:02d}"
                good = [r for r in h if abs(r["purge_step_mol"] / r["purge_target_mol"] - 1) < 0.05
                        and abs(r["params"]["h_al"] - f["params"]["h_al"]) < 1e-6]
                sl = slope
                if len(good) >= 2 and abs(good[-1]["params"]["h_cms"] - good[-2]["params"]["h_cms"]) > 1e-4:
                    sl = ((math.log(good[-1]["O2_ppm"]) - math.log(good[-2]["O2_ppm"]))
                          / (good[-1]["params"]["h_cms"] - good[-2]["params"]["h_cms"]))
                sl = min(sl, -1.0)
                params = dict(f["params"])
                params["cv_purge"] = params["cv_purge"] * f["purge_target_mol"] / f["purge_step_mol"]
                hc = params["h_cms"] + (math.log(4500.0) - math.log(f["O2_ppm"])) / sl
                params["h_cms"] = round(min(max(hc, 0.5 * params["h_cms"]), 1.6 * params["h_cms"]), 4)
            run(PY + [HERE / "sizing" / "size_bed.py", "--start", json.dumps(params), "--warm", warm, "--out", out,
                      "--max-iter", "4", "--tol", "5e-5", "--max-maps", "40", "--trace-src", TRACE_SRC,
                      "--slope", f"{min(sl, -1.0):.3f}"], HERE / "sizing")
    if not js.exists():
        log(f"sizing round {out} ended without its json")
        sys.exit(1)
    cur = json.loads(js.read_text())
    log(f"sizing round {out}: final {cur['final']['params']}, O2 {cur['final']['O2_ppm']:.0f} ppm, sized {sized(cur['final'])}")
    if sized(cur["final"]):
        break
    rounds.append(cur)
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
