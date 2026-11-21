"""Gas molecules drawn in the beds (env viz): counts and species from the solver, cached history.

Origin: psa-viz column/render/scene.py GasDots + render_column.py precompute_dots (commit 709af6f),
generalized to five species and two beds. Molecules live in a slab just behind the cut face:
* the number per height bin follows the local total gas concentration (the bed fills on
  pressurization and empties in blowdown) and the species follow the local mole fractions;
  trace species are drawn at exaggerated counts, n_i ~ C y_i w_i with the weights w_i of the
  config (stated in the footnote);
* they drift with the solver's interstitial gas velocity (u / eps of each cell), mapped to a
  watchable video speed, v = vmax tanh(k u_int / vmax) bed lengths per video second (monotonic in
  the real velocity, not to scale);
* births and deaths happen only when a bin leaves a +-band around its target (no flicker).
The history is computed once on the 30 fps timeline (one warm-up loop, then one stored loop) and
cached in render/cache/, so drafts (every second frame) and finals show the same molecules.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent


class GasDots:
    def __init__(self, pack, n_species, weights, n_ref=1100, r_dot=0.16, y0=-0.7, y1=1.2, nbins=40, seed=3, cut_y=0.0):
        self.p, self.n_ref, self.r_dot = pack, n_ref, r_dot
        self.w = np.asarray(weights, float)
        self.ns = n_species
        self.y0, self.y1, self.nbins = y0, y1, nbins
        self.rng = np.random.default_rng(seed)
        self.edges = np.linspace(0.0, pack.H, nbins + 1)
        self.pos = np.zeros((0, 3))
        self.sp = np.zeros(0, int)
        n = 400_000
        rho = (pack.R - 0.3) * np.sqrt(self.rng.random(n))
        ph = self.rng.random(n) * 2 * np.pi
        pts = np.c_[rho * np.cos(ph), rho * np.sin(ph), self.rng.random(n) * pack.H]
        pts = pts[(pts[:, 1] > y0) & (pts[:, 1] < y1)]
        shown = cKDTree(pack.pos[pack.pos[:, 1] > cut_y])
        dist, _ = shown.query(pts)
        self.pool = pts[dist > pack.r + r_dot * 0.9]
        pool_bin = np.clip(np.digitize(self.pool[:, 2], self.edges) - 1, 0, nbins - 1)
        self.pool_by_bin = [np.nonzero(pool_bin == b)[0] for b in range(nbins)]

    def targets(self, C, y, z_cells, C_ref):
        """Target counts per (species, bin) from the solver's C(z) and y_i(z)."""
        zc = 0.5 * (self.edges[1:] + self.edges[:-1]) / self.p.H
        Cb = np.interp(zc, z_cells, C) / C_ref
        yb = np.array([np.interp(zc, z_cells, y[i]) for i in range(self.ns)])
        return self.n_ref * Cb[None] * yb * self.w[:, None] / self.nbins

    def _spawn(self, b, k, species):
        idx = self.rng.choice(self.pool_by_bin[b], size=k, replace=len(self.pool_by_bin[b]) < k)
        return self.pool[idx] + self.rng.normal(0, 0.05, (k, 3)), np.full(k, species)

    def match(self, tgt, band=0.08):
        b = np.clip(np.digitize(self.pos[:, 2], self.edges) - 1, 0, self.nbins - 1)
        keep = np.ones(len(self.pos), bool)
        new_p, new_s = [], []
        for bb in range(self.nbins):
            for s in range(self.ns):
                t = tgt[s, bb]
                tol = max(1.0, band * t)
                have = np.nonzero((b == bb) & (self.sp == s) & keep)[0]
                if len(have) > t + tol:
                    keep[self.rng.choice(have, len(have) - int(np.ceil(t + tol * 0.5)), replace=False)] = False
                elif len(have) < t - tol:
                    k = int(np.floor(t - tol * 0.5)) - len(have)
                    if k > 0:
                        p, sp = self._spawn(bb, k, s)
                        new_p.append(p)
                        new_s.append(sp)
        self.pos = np.concatenate([self.pos[keep]] + new_p) if new_p else self.pos[keep]
        self.sp = np.concatenate([self.sp[keep]] + new_s) if new_s else self.sp[keep]

    def advance(self, u_faces, eps_cells, z_faces, z_cells, tgt, dt_video, k=0.25, vmax=0.6, jitter=0.025):
        u_cells = 0.5 * (u_faces[1:] + u_faces[:-1]) / eps_cells  # interstitial, m/s
        u = np.interp(self.pos[:, 2] / self.p.H, z_cells, u_cells)
        v = vmax * np.tanh(k * u / vmax)
        self.pos[:, 2] += v * dt_video * self.p.H
        self.pos += self.rng.normal(0, jitter, self.pos.shape)
        ok = (self.pos[:, 2] > 0) & (self.pos[:, 2] < self.p.H)
        r = np.hypot(self.pos[:, 0], self.pos[:, 1])
        ok &= (r < self.p.R - 0.2) & (self.pos[:, 1] > self.y0) & (self.pos[:, 1] < self.y1)
        self.pos, self.sp = self.pos[ok], self.sp[ok]
        self.match(tgt)


def molecule_history(frames, pack, eps_cells, weights, n_ref, key_extra="", cache_dir=HERE / "cache", log=print):
    """Positions/species of the molecules of both beds at every frame of the 30 fps timeline (cached)."""
    nf = len(frames["t_video"])
    key = hashlib.sha256((json.dumps([list(map(float, weights)), n_ref]) + key_extra).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"molecules_{key}.npz"
    if path.exists():
        return dict(np.load(path))
    z_cells, z_faces = frames["z"], frames["z_faces"]
    C_ref = float(max(frames["C0"].max(), frames["C1"].max()))
    out = {}
    for k in (0, 1):
        dots = GasDots(pack, len(weights), weights, n_ref=n_ref, seed=3 + k)
        tgt = dots.targets(frames[f"C{k}"][0], frames[f"y{k}"][0], z_cells, C_ref)
        dots.match(tgt)
        pos, sp, off = [], [], [0]
        for f in range(2 * nf):
            i = f % nf
            dt = frames["t_video"][1] - frames["t_video"][0]
            tgt = dots.targets(frames[f"C{k}"][i], frames[f"y{k}"][i], z_cells, C_ref)
            dots.advance(frames[f"u{k}"][i], eps_cells, z_faces, z_cells, tgt, dt)
            if f >= nf:
                pos.append(dots.pos.astype(np.float32))
                sp.append(dots.sp.astype(np.int8))
                off.append(off[-1] + len(dots.pos))
        out[f"pos{k}"] = np.concatenate(pos)
        out[f"sp{k}"] = np.concatenate(sp)
        out[f"off{k}"] = np.array(off)
    out["r_dot"] = np.array(0.16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    log(f"molecule history written: {path.name}")
    return dict(np.load(path))
