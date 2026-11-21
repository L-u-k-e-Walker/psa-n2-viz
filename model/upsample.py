"""Upsample a saved CSS state to a finer grid with the same geometry (env viz; numpy only).

usage: python upsample.py SRC_DIR DST.npy '{"n_al": 30, "n_cms": 180, ...case overrides}'
Writes the refined state (every cell repeated) for use as a warm start, and DST.json with the
case arguments of the refined case.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from psan2viz import css  # noqa: E402
from psan2viz.cases import case_from_args  # noqa: E402
from psan2viz.plant import Plant  # noqa: E402

src, dst, over = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])
d = np.load(src / "css_state.npz", allow_pickle=False)
args = json.loads(str(d["case_args"]))
_, pl1, x1 = css.load(src)
args2 = dict(args, **over)
pl2 = Plant(case_from_args(args2))
fa, fc = args2["n_al"] // args["n_al"], args2["n_cms"] // args["n_cms"]
assert args2["n_al"] == fa * args["n_al"] and args2["n_cms"] == fc * args["n_cms"]
reps = np.concatenate([np.full(args["n_al"], fa), np.full(args["n_cms"], fc)])
x2 = np.zeros(pl2.size)
for k in (0, 1):
    c1, q1 = pl1.bed(x1, k)
    c2, q2 = pl2.bed(x2, k)
    c2[:] = np.repeat(c1, reps, axis=1)
    q2[:] = np.repeat(q1, reps, axis=1)
x2[pl2.s_t] = x1[pl1.s_t]
np.save(dst, x2)
args2.pop("solver_params", None)
Path(str(dst).replace(".npy", ".json")).write_text(json.dumps(args2, indent=1))
print(f"upsampled {pl1.size} -> {pl2.size} states: {dst}")
