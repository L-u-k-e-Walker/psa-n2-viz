"""Which state blocks carry the CSS residual, and how fast does each decay? (env viz)

usage: python residual_diag.py STATE_DIR_or_.npy [maps]
Runs plain swapped half-cycle maps from the state and prints, per block (bed part x species, receiver),
the RMS of the normalized half-map residual and its full-cycle decay factor (ratio of the RMS
over two consecutive full cycles). Diagnostics for the CSS solver only.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from psan2viz import css  # noqa: E402
from psan2viz.cases import n2_generator_case  # noqa: E402
from psan2viz.plant import Plant  # noqa: E402

src = Path(sys.argv[1])
n_maps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
if src.suffix == ".npy":
    import json
    extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    pl = Plant(n2_generator_case(rtol=1e-6, atol_rel=1e-9, atol_rel_trace=1e-6, **extra))
    x = np.load(src)
else:
    _, pl, x = css.load(src)
xs = [x]
for _ in range(n_maps):
    xs.append(pl.half_map(xs[-1])[0])
blocks = [pl.residual_breakdown(xs[i], xs[i + 1]) for i in range(n_maps)]
keys = sorted(blocks[0], key=lambda k: -blocks[-1][k])
print(f"{'block':20s} " + " ".join(f"map{i + 1:>2d}" for i in range(n_maps)) + "   decay/cycle")
for k in keys[:14]:
    v = [b[k] for b in blocks]
    dec = v[-1] / v[-3] if v[-3] > 0 else float("nan")
    print(f"{k:20s} " + " ".join(f"{a:.1e}" for a in v) + f"   {dec:.3f}")
tot = [float(np.sqrt(np.mean(((xs[i + 1] - xs[i]) / pl.scale)[:pl.n_phys] ** 2))) for i in range(n_maps)]
print("total RMS", " ".join(f"{t:.2e}" for t in tot))
