"""Gate 3: the plotted values come straight from the solver (env viz). Writes docs/gate3_spotcheck.json.

For three frames of the render timeline, the values the panels draw (profiles, loadings,
pressures, receiver composition, valve flows) are read from the physics cache (render/frames.py)
and compared with an independent re-integration of the CSS cycle at the same cycle times.
Also checks the loop seam: the state at the end of the stored cycle equals the start.
Run from the repo root: conda run -n viz python -I tools/gate3_spotcheck.py
"""
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "render")]
from frames import CSS_DIR, frame_cache  # noqa: E402
from psan2viz import css  # noqa: E402
from psan2viz.analysis import Cycle  # noqa: E402

cfg = tomllib.loads((ROOT / "render" / "config.toml").read_text())
d, tm, _ = frame_cache(cfg["timing"])
case, pl, x0 = css.load(CSS_DIR)
x_end, recs = pl.run_cycle(x0)  # independent re-integration
cyc = Cycle(pl, recs)
picks = [int(len(d["t_video"]) * f) for f in (0.07, 0.41, 0.83)]
diffs = {}
for f in picks:
    fr = cyc.frame(float(d["t_cycle"][f]))
    worst = 0.0
    for key in ("y0", "y1", "q0", "q1", "y_tank"):
        worst = max(worst, float(np.max(np.abs(d[key][f] - fr[key]))))
    for key in ("P0", "P1", "p_tank", "F0_feed", "F1_feed", "F0_product", "F1_product", "F0_vent", "F1_vent",
                "F0_purge", "F1_purge", "F0_eq_top", "F0_eq_bot"):
        worst = max(worst, abs(float(d[key][f]) - float(fr[key])))
    diffs[f"frame {f} (t = {d['t_cycle'][f]:.2f} s)"] = worst
rms, mx = pl.norm_change(x_end, x0)
out = dict(frames=diffs, max_abs_diff=max(diffs.values()), limit=1e-6, loop_seam_rms=rms, loop_seam_max=mx)
out["pass"] = out["max_abs_diff"] < 1e-6 and mx < 1e-4
(ROOT / "docs" / "gate3_spotcheck.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
