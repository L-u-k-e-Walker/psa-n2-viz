"""Solve, verify, save and load the cyclic steady state of the plant (numpy + scipy only)."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .cases import case_from_args, n2_generator_case
from .plant import Plant


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def solve(case=None, x0=None, tol=1e-8, log=None, **case_kw):
    """CSS of a case (built from ``case_kw`` when ``case`` is None). Returns a result dict.

    The fixed point of the swapped half-cycle map is found with Anderson acceleration, then
    one plain full cycle verifies it (cycle-to-cycle change, the G1b quantity)."""
    case = case or n2_generator_case(**case_kw)
    pl = Plant(case)
    t0 = time.time()

    def progress(h):
        if log:
            log(f"  iter {h['iter']:3d}  rms {h['rms']:.3e}  max {h['max']:.3e}  {h['wall_s']:.0f}s")

    x, hist = pl.run_to_css(x0=x0, tol=tol, progress=progress)
    t_css = time.time() - t0
    x_end, recs = pl.run_cycle(x)
    rms, mx = pl.norm_change(x_end, x)
    met = pl.cycle_metrics(recs)
    met.update(css_change_rms=rms, css_change_max=mx, css_iterations=len(hist), wall_css_s=t_css,
               wall_verify_cycle_s=time.time() - t0 - t_css)
    return dict(case=case, plant=pl, x=x, recs=recs, hist=hist, metrics=met)


def save(res, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "css_state.npz", x_start=res["x"],
             case_args=json.dumps(res["case"].case_args, default=_jsonable))
    (folder / "metrics.json").write_text(json.dumps(res["metrics"], indent=1, default=_jsonable))
    with open(folder / "css_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "rms", "max", "wall_s"])
        for h in res["hist"]:
            w.writerow([h["iter"], h["rms"], h["max"], round(h["wall_s"], 1)])


def load(folder):
    """(case, plant, x_start) from a saved CSS state."""
    d = np.load(Path(folder) / "css_state.npz", allow_pickle=False)
    case = case_from_args(json.loads(str(d["case_args"])))
    return case, Plant(case), d["x_start"]
