"""Carry converged trace profiles (CO2, H2O) from one CSS state into another (env viz; numpy only).

Grid sequencing for the slow trace profiles: they are converged cheaply on a coarse grid (frozen-bulk
march, model/css/trace_css.py) and carried into the fine-grid state, whose bulk is kept. Positions map
layer by layer: in the alumina by height above the bottom scaled to the layer (z / h_al), in the CMS by
the relative position in the CMS layer ((z - h_al) / h_cms), so the source may also have a different
CMS height. Gas: the trace mole fractions are carried (times the target's local total concentration);
loadings are carried as they are; linear interpolation between cell centres. The receiver's trace
fractions are carried too. The result is only a warm start: the target is converged by full-model
cycles afterwards.

usage: python transfer_trace.py TRACE_SRC_DIR TARGET_DIR OUT.npy
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from psan2viz import css  # noqa: E402
from psan2viz.trace import transfer  # noqa: E402


src, dst, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
case1, pl1, x1 = css.load(src)
case2, pl2, x2 = css.load(dst)
x = transfer(pl1, x1, pl2, x2)
np.save(out, x)
d = (x[pl2.trace_indices()] - x2[pl2.trace_indices()]) / pl2.scale[pl2.trace_indices()]
print(f"trace profiles {src} -> {dst}: {out}; normalized change rms {np.sqrt(np.mean(d * d)):.3e}, max {np.abs(d).max():.3e}")
