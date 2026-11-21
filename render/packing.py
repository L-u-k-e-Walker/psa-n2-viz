"""Random packing of equal spheres in a cylinder for the schematic 3D beds (env viz).

Origin: psa-viz column/render/packing.py (git@github.com:L-u-k-e-Walker/psa-viz.git, commit 709af6f),
unchanged method: soft-sphere relaxation under gravity (harmonic overlap forces, overdamped),
spheres start at random overlapping positions in a tall cylinder, gravity settles them, then
gravity is switched off to relax and the bed is cut at the target height. Changed: the bed is
taller (aspect 4.1, matching the bed boxes of every layout) and the seed. Units: particle
diameter = 1. Geometry is schematic (docs/INPUTS.md): D/d = 14, far coarser than a real bed,
L/D ~4 instead of ~7. Both beds use this packing (bed 2 mirrored). The alumina layer is the
bottom z_al fraction of the particles, coloured with its own ramp.
Writes packing.npz (pos[N,3], radius, R_col, H_bed, seed). Run: conda run -n viz python -I packing.py
"""
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).parent
R_COL, ASPECT = 7.0, 4.1
H_BED = 2 * R_COL * ASPECT
SEED = 11
rng = np.random.default_rng(SEED)

n0 = int(0.62 * np.pi * R_COL**2 * H_BED / (np.pi / 6) * 0.95)
h0 = 1.7 * H_BED
rho = (R_COL - 0.5) * np.sqrt(rng.random(n0))
phi = rng.random(n0) * 2 * np.pi
x = np.c_[rho * np.cos(phi), rho * np.sin(phi), 0.5 + rng.random(n0) * (h0 - 1)]

t0 = time.time()
k, g, dt, steps = 1.0, 0.008, 0.25, 26000
for it in range(steps):
    tree = cKDTree(x)
    pairs = tree.query_pairs(1.0, output_type="ndarray")
    d = x[pairs[:, 1]] - x[pairs[:, 0]]
    r = np.linalg.norm(d, axis=1)
    ov = 1.0 - r
    f = (k * ov / np.maximum(r, 1e-9))[:, None] * d
    F = np.zeros_like(x)
    for a in range(3):
        F[:, a] -= np.bincount(pairs[:, 0], f[:, a], len(x))
        F[:, a] += np.bincount(pairs[:, 1], f[:, a], len(x))
    rr = np.hypot(x[:, 0], x[:, 1])
    ow = rr + 0.5 - R_COL
    m = ow > 0
    F[m, 0] -= k * ow[m] * x[m, 0] / rr[m]
    F[m, 1] -= k * ow[m] * x[m, 1] / rr[m]
    of = 0.5 - x[:, 2]
    F[of > 0, 2] += k * of[of > 0]
    gg = g if it < steps - 9000 else g * max(0.0, 1 - (it - (steps - 9000)) / 2000)  # switch gravity off to relax
    F[:, 2] -= gg
    step = dt * F
    sn = np.linalg.norm(step, axis=1)
    step *= np.minimum(1.0, 0.05 / np.maximum(sn, 1e-12))[:, None]
    x += step
    if it % 2500 == 0 or it == steps - 1:
        print(f"it {it} max overlap {ov.max() if ov.size else 0:.3f} top {x[:, 2].max():.1f} t {time.time()-t0:.0f}s", flush=True)

keep = x[:, 2] <= H_BED - 0.5
x = x[keep]
tree = cKDTree(x)
pairs = tree.query_pairs(1.0, output_type="ndarray")
ovl = 1 - np.linalg.norm(x[pairs[:, 1]] - x[pairs[:, 0]], axis=1)
core = (np.hypot(x[:, 0], x[:, 1]) < R_COL - 2) & (x[:, 2] > 2) & (x[:, 2] < H_BED - 2)
v_core = np.pi * (R_COL - 2) ** 2 * (H_BED - 4)
phi_core = core.sum() * np.pi / 6 / v_core
h_bed = float(min(H_BED, x[:, 2].max() + 0.5))
np.savez(HERE / "packing.npz", pos=x, radius=0.5, R_col=R_COL, H_bed=h_bed, seed=SEED)
print(f"H_bed = {h_bed:.2f} (L/D = {h_bed / (2 * R_COL):.2f}), N = {len(x)}, max overlap {ovl.max():.3f} d, "
      f"mean overlap {ovl.mean():.4f} d, core packing fraction {phi_core:.3f}")
