"""Sampling and derived quantities of one stored cycle (used by the gates and the renderer).

``Cycle`` wraps the records of one plain cycle (``Plant.run_cycle``): the state at any time
in the cycle comes from the BDF dense output of its segment, and everything drawn on screen
(profiles, pressures, valve flows, readouts) is derived from that state here.
"""
from __future__ import annotations

import numpy as np

from .plant import VALVES_BOTTOM, VALVES_TOP


class Cycle:
    def __init__(self, plant, recs):
        self.pl = plant
        self.recs = recs
        self.case = plant.case
        self.dur = np.array([r["duration"] for r in recs])
        self.starts = np.concatenate([[0.0], np.cumsum(self.dur)])
        self.T = float(self.starts[-1])
        self.segments = self.case.segments()
        self.sp = list(self.case.species)
        g = plant.g
        self.z = g.z / g.length  # cell centres, z/L
        self.z_al = sum(lay.height for lay in g.layers if lay.name == "alumina") / g.length

    def seg_index(self, t):
        return int(min(np.searchsorted(self.starts, t, side="right") - 1, len(self.dur) - 1))

    def state(self, t):
        """Full state vector at cycle time t (0 <= t <= T)."""
        t = float(np.clip(t, 0.0, self.T))
        s = self.seg_index(t)
        return self.recs[s]["sol"].sol(min(t - self.starts[s], self.dur[s])), s

    def frame(self, t):
        """Everything the visual needs at cycle time t, as a dict of numpy arrays / floats."""
        x, s = self.state(t)
        pl = self.pl
        seg = self.segments[s]
        X = x[:, None]
        flows, beds, C, PT, yT, _ = pl._valve_flows(X, seg)
        out = dict(t=float(t), frac=float(t / self.T), seg=s, seg_name=seg.name, t_in_seg=float(t - self.starts[s]),
                   seg_frac=float((t - self.starts[s]) / self.dur[s]), p_tank=float(PT[0]), y_tank=yT[:, 0].copy())
        for k in (0, 1):
            c, q = beds[k]
            Ck = C[k][:, 0]
            out[f"y{k}"] = (c[:, :, 0] / Ck)  # (n, N)
            out[f"q{k}"] = q[:, :, 0].copy()
            out[f"C{k}"] = Ck.copy()
            out[f"P{k}"] = float(Ck.mean() * pl.RT / 1e5)
            for v in VALVES_BOTTOM + VALVES_TOP:
                F = flows.get((k, v))
                out[f"F{k}_{v}"] = float(F[0][0]) if F is not None else 0.0  # mol/s INTO bed k (0 if closed)
                out[f"open{k}_{v}"] = F is not None
        out["F_product_out"] = float(pl.product_withdrawal(PT)[0])
        return out

    def sample(self, n_per_seg=40):
        ts = np.concatenate([self.starts[i] + np.linspace(0, d, n_per_seg, endpoint=False) for i, d in enumerate(self.dur)]
                            + [[self.T]])
        return ts, [self.frame(t) for t in ts]

    # -------------------------------------------------------------- checks
    def h2o_front(self, frames, frac=0.01):
        """Highest z/L where y_H2O >= frac * y_H2O(feed), over the sampled frames and both beds,
        and the largest H2O loading on the CMS relative to its feed-equilibrium value."""
        iw = self.sp.index("H2O")
        yf = self.case.y_feed[iw]
        g = self.pl.g
        cms = g.layer_of == [i for i, lay in enumerate(g.layers) if lay.name == "CMS"][0]
        z_front, q_cms = 0.0, 0.0
        for fr in frames:
            for k in (0, 1):
                wet = np.nonzero(fr[f"y{k}"][iw] >= frac * yf)[0]
                if wet.size:
                    z_front = max(z_front, float(g.z_faces[wet.max() + 1] / g.length))
                q_cms = max(q_cms, float(np.max(fr[f"q{k}"][iw, cms] / self.pl.q_scale[iw, cms])))
        return z_front, q_cms

    def eq_residual(self):
        """|P_hi - P_lo| at the end of each equalization / at its start."""
        res = []
        for i, seg in enumerate(self.segments):
            if seg.name != "EQ":
                continue
            p0 = [self.pl.bed_pressure(self.recs[i]["x0"], k) for k in (0, 1)]
            p1 = [self.pl.bed_pressure(self.recs[i]["x_end"], k) for k in (0, 1)]
            res.append(dict(seg=i, dP_start=abs(p0[0] - p0[1]), dP_end=abs(p1[0] - p1[1]),
                            ratio=abs(p1[0] - p1[1]) / abs(p0[0] - p0[1]), P_end=p1))
        return res

    def symmetry(self, n=200):
        """max |bed 1 at t - bed 0 at t + T/2| over the half-cycle, per-species scaled."""
        pl = self.pl
        sc = pl.scale[pl.s_bed[0]]
        worst = 0.0
        for t in np.linspace(0, self.T / 2, n, endpoint=False):
            x1, _ = self.state(t)
            x0, _ = self.state(t + self.T / 2)
            worst = max(worst, float(np.max(np.abs(x1[pl.s_bed[1]] - x0[pl.s_bed[0]]) / sc)))
        return worst
