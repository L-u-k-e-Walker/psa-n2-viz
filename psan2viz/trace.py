"""Frozen-bulk march of the trace profiles (CO2, H2O) to their cyclic steady state.

The water and CO2 profiles relax over ~1e4 cycles (the alumina layer slowly fills or dries until the
water fed per cycle equals the water vented), while the bulk (pressures, flows, N2/O2/Ar) settles in
tens. The trace species barely act back on the bulk (water is 915 ppm of the feed), so their slow
march can be integrated with the bulk frozen on one recorded full cycle of the complete model:

    dz/dt = f(t, X_bulk(t) with X[T] = z)[T]

where T are the trace entries (gas and loading of CO2 and H2O in both beds, the receiver), f is the
plant's own right-hand side (same physics, same valve and boundary couplings: feed, equalization
between the beds, the receiver) and X_bulk(t) is the dense output of that recorded cycle. A trace
cycle costs a small fraction of a full cycle (small sparse implicit system). The bulk is refreshed
with full cycles of the complete model between marches, and the final state is always verified by
full-model cycles (the frozen bulk is an acceleration device, not part of the model).
"""
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix


class TraceMarch:
    def __init__(self, plant, recs, rtol=None, thr=1e-4):
        """plant: Plant; recs: records of one full cycle of the complete model (Plant.run_cycle)."""
        self.pl = plant
        self.recs = recs
        self.segs = plant.case.segments()
        assert len(self.segs) == len(recs)
        self.T = plant.trace_indices()
        self.sT = plant.scale[self.T]
        self.rtol = rtol if rtol is not None else plant.case.rtol
        self.atol = plant.atol[self.T]
        self.S = self._sparsity(thr)

    def _X(self, i, t, Z):
        X = self.recs[i]["sol"].sol(t)  # bulk at time t (full state)
        if Z.ndim == 1:
            X = X.copy()
            X[self.T] = Z
            return X
        X = np.repeat(X[:, None], Z.shape[1], axis=1)
        X[self.T] = Z
        return X

    def f(self, i, t, Z):
        return self.pl.rhs(t, self._X(i, t, Z), self.segs[i])[self.T]

    def _sparsity(self, thr):
        """Pattern of the trace Jacobian, detected numerically (one vectorized FD evaluation per
        segment, at three times each), entries below thr x the row's largest dropped (the weak
        long-range coupling through the velocity field)."""
        nT = self.T.size
        pat = np.zeros((nT, nT), bool)
        for i, rec in enumerate(self.recs):
            for frac in (0.02, 0.3, 0.9):
                t = frac * rec["duration"]
                z = rec["sol"].sol(t)[self.T]
                h = 1e-6 * np.maximum(np.abs(z), self.sT)
                Z = z[:, None] + np.diag(h)
                f0 = self.f(i, t, z)
                J = (self.f(i, t, Z) - f0[:, None]) / h[None, :]
                row = np.abs(J).max(axis=1, keepdims=True)
                pat |= np.abs(J) > thr * np.maximum(row, 1e-300)
        pat |= np.eye(nT, dtype=bool)
        return csr_matrix(pat.astype(float))

    def cycle(self, z0):
        """One full cycle of the trace subsystem on the frozen bulk."""
        z = z0.copy()
        for i, rec in enumerate(self.recs):
            sol = solve_ivp(lambda t, Z, i=i: self.f(i, t, Z), (0.0, rec["duration"]), z, method="BDF",
                            rtol=self.rtol, atol=self.atol, jac_sparsity=self.S, vectorized=True)
            if sol.status < 0:
                raise RuntimeError(f"trace segment {i}: {sol.message}")
            z = np.maximum(sol.y[:, -1], 0.0)
        return z

    def march(self, z0, tol=1e-7, max_cycles=5000, report=200, progress=None):
        """Plain trace cycles until the RMS of the normalized per-cycle change < tol."""
        z = z0.copy()
        t0 = time.time()
        rms = np.inf
        for k in range(1, max_cycles + 1):
            zn = self.cycle(z)
            d = (zn - z) / self.sT
            rms = float(np.sqrt(np.mean(d * d)))
            z = zn
            if progress and (k % report == 0 or rms < tol or k == 1):
                progress(dict(cycle=k, rms=rms, max=float(np.abs(d).max()), wall_s=time.time() - t0))
            if rms < tol:
                break
        return z, k, rms


def converge(plant, x0, tol=1e-6, march_tol=None, march_max=3000, full_between=3, outer_max=12,
             rtol_trace=None, progress=None, checkpoint=None):
    """Alternate (a) ``full_between`` full-model cycles (bulk follows the trace state) and (b) a
    frozen-bulk trace march on the last recorded cycle, until a full-model cycle changes the whole
    normalized state by less than ``tol`` (RMS). Returns (x, history)."""
    pl = plant
    P = pl.n_phys
    s = pl.scale[:P]
    march_tol = march_tol if march_tol is not None else 0.1 * tol
    x = x0.copy()
    hist = []
    t0 = time.time()
    for outer in range(1, outer_max + 1):
        for _ in range(full_between):
            x_end, recs = pl.run_cycle(x)
            d = (x_end[:P] - x[:P]) / s
            rms_full = float(np.sqrt(np.mean(d * d)))
            x = pl.project(x_end)
        hist.append(dict(iter=len(hist) + 1, kind="full", rms=rms_full, max=float(np.abs(d).max()),
                         wall_s=time.time() - t0))
        if progress:
            progress(dict(hist[-1], note=f"outer {outer}"))
        if rms_full < tol:
            return x, hist
        tm = TraceMarch(pl, recs, rtol=rtol_trace)
        tm_start = time.time()
        z, n_cyc, rms_t = tm.march(x[tm.T], tol=march_tol, max_cycles=march_max,
                                   progress=(lambda h: progress(dict(h, kind="trace", iter=len(hist) + 1,
                                                                     note=f"outer {outer}")) if progress else None))
        x = x.copy()
        x[tm.T] = z
        hist.append(dict(iter=len(hist) + 1, kind="march", rms=rms_t, max=float("nan"), wall_s=time.time() - t0))
        if progress:
            progress(dict(hist[-1], note=f"outer {outer}: {n_cyc} trace cycles in {time.time() - tm_start:.0f} s, "
                                         f"pattern nnz {tm.S.nnz} of {tm.T.size}^2"))
        if checkpoint:
            checkpoint(x)
    return x, hist


def layer_coord(plant):
    """Cell centres as layer-relative positions: 0..1 in the alumina (z / h_al), 1..2 in the CMS."""
    a = plant.case.case_args
    z = plant.g.z
    h_al, h_cms = a["h_al"], a["h_cms"]
    return np.where(z < h_al, z / h_al, 1.0 + (z - h_al) / h_cms)


def transfer(pl_src, x_src, pl_dst, x_dst):
    """x_dst with its trace profiles (CO2, H2O) replaced by those of x_src, mapped layer by layer
    (grid and CMS height may differ): gas as mole fractions times the target's total concentration,
    loadings as they are, linear interpolation between cell centres; receiver trace fractions too.
    A warm start only: the target is converged by full-model cycles afterwards."""
    sp = list(pl_dst.case.species)
    assert sp == list(pl_src.case.species)
    tr = [sp.index("CO2"), sp.index("H2O")]
    s1, s2 = layer_coord(pl_src), layer_coord(pl_dst)
    x = x_dst.copy()
    for k in (0, 1):
        c1, q1 = pl_src.bed(x_src, k)
        c2, q2 = pl_dst.bed(x, k)
        C1, C2 = c1.sum(0), c2.sum(0)
        for i in tr:
            c2[i] = np.interp(s2, s1, c1[i] / C1) * C2
            q2[i] = np.interp(s2, s1, q1[i])
    t1, t2 = x_src[pl_src.s_t], x[pl_dst.s_t].copy()
    for i in tr:
        t2[i] = t1[i] / t1.sum() * t2.sum()
    x[pl_dst.s_t] = t2
    return x
