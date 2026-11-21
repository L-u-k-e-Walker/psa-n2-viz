"""psa-viz's single-column 4-step model, generalized to n components and layered beds.

Origin: psa-viz ``psaviz/column.py`` (``ColumnModel``; commit 709af6f). Same steps and
logic (RP / FEED / BD / PURGE, valve and free ends, raffinate tank with pre-fill),
rewritten on the shared kernel ``psan2viz.bed.BedGrid.rates`` so that the regression
gate G1c checks the kernel the two-bed plant uses. Component order is free; the
"light" (raffinate) and "heavy" (extract) roles are given by index.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .bed import BedGrid


@dataclass
class Step:
    kind: str  # RP, FEED, BD, PURGE
    duration: float  # s
    cv: float  # mol/(s bar)


@dataclass
class ColumnCase:
    grid: BedGrid
    y_feed: np.ndarray  # (n,)
    p_feed: float  # bar
    p_vent: float
    tank_volume: float  # m^3
    tank_p_set: float  # bar
    p_init: float
    y_init: np.ndarray  # (n,) initial column and tank gas
    steps: list
    light: int  # index of the raffinate (purity/recovery) component
    heavy: int
    rtol: float = 1e-6
    atol_rel: float = 1e-9
    css_tol: float = 1e-7
    max_cycles: int = 400
    css_components: tuple | None = None  # components entering the CSS norm (None = all)
    provenance: dict = field(default_factory=dict)


class ColumnN:
    def __init__(self, case: ColumnCase):
        self.case = case
        g = self.g = case.grid
        n, N = self.n, self.N = g.n, g.N
        self.yf = np.asarray(case.y_feed, float)
        self.s_c = slice(0, n * N)
        self.s_q = slice(n * N, 2 * n * N)
        self.s_t = slice(2 * n * N, 2 * n * N + n)
        self.s_cnt = slice(2 * n * N + n, 2 * n * N + 5 * n)
        self.size = 2 * n * N + 5 * n
        self.c_ref = case.p_feed * 1e5 / g.RT
        self.q_ref = float(g.qs.sum(1).max())
        self.n_ref = self.c_ref * g.sum_epsdz * g.area
        atol = np.empty(self.size)
        atol[self.s_c] = case.atol_rel * self.c_ref
        atol[self.s_q] = case.atol_rel * self.q_ref
        atol[self.s_t] = case.atol_rel * self.c_ref
        atol[self.s_cnt] = case.atol_rel * self.n_ref
        self.atol = atol

    def split(self, x):
        n, N = self.n, self.N
        tail = x.shape[1:]
        return (x[self.s_c].reshape((n, N) + tail), x[self.s_q].reshape((n, N) + tail),
                x[self.s_t], x[self.s_cnt])

    def initial_state(self):
        c = self.case
        x = np.zeros(self.size)
        cc, qq, cR, _ = self.split(x)
        C0 = c.p_init * 1e5 / self.g.RT
        cc[:] = (np.asarray(c.y_init) * C0)[:, None]
        qq[:] = self.g.equilibrium(cc[:, :, None])[:, :, 0]
        cR[:] = np.asarray(c.y_init) * c.tank_p_set * 1e5 / self.g.RT
        return x

    def tank_pressure(self, x):
        return x[self.s_t].sum(0) * self.g.RT / 1e5

    def column_pressure(self, x):
        c, _, _, _ = self.split(x)
        return c.sum(0).mean(0) * self.g.RT / 1e5

    def inventory(self, x):
        c, q, cR, _ = self.split(x)
        return self.g.inventory(c, q) + self.case.tank_volume * cR

    # ------------------------------------------------------------------- rhs
    def _ends(self, c, C, cR, kind, cv):
        RT, K = self.g.RT, c.shape[2]
        P0 = C[0] * RT / 1e5
        PN = C[-1] * RT / 1e5
        Pt = cR.sum(0) * RT / 1e5
        yR = cR / cR.sum(0)
        y0 = c[:, 0] / C[0]
        yN = c[:, -1] / C[-1]
        yf = np.broadcast_to(self.yf[:, None], (self.n, K))
        closed = ("valves", np.zeros(K), np.zeros((self.n, K)))
        if kind in ("RP", "BD", "FEED"):
            p_up = self.case.p_vent if kind == "BD" else self.case.p_feed
            F, Nc = self.g.end_flows([(cv * (p_up - P0), yf)], y0)
            bot = ("valves", F, Nc)
            top = ("free", yR) if kind == "FEED" else closed
        elif kind == "PURGE":
            F, Nc = self.g.end_flows([(cv * (Pt - PN), yR)], yN)
            top = ("valves", F, Nc)
            bot = ("free", yf)
        else:
            raise ValueError(kind)
        return bot, top, yR

    def rhs(self, t, x, kind, cv, acc):
        vec = x.ndim == 2
        X = x if vec else x[:, None]
        c, q, cR, _ = self.split(X)
        C = c.sum(0)
        bot, top, yR = self._ends(c, C, cR, kind, cv)
        dc, dq, u, f_bot, f_top = self.g.rates(c, q, C, bot, top)
        K = X.shape[1]
        out = np.empty_like(X)
        out[self.s_c] = dc.reshape(self.n * self.N, K)
        out[self.s_q] = dq.reshape(self.n * self.N, K)
        ex = f_top if kind in ("FEED", "PURGE") else np.zeros((self.n, K))
        prod = np.zeros((self.n, K)) if acc else np.maximum(0.0, ex.sum(0)) * yR
        out[self.s_t] = (ex - prod) / self.case.tank_volume
        cnt = np.zeros((4 * self.n, K))
        n = self.n
        if kind in ("RP", "FEED"):
            cnt[0:n] = f_bot
        else:
            cnt[n:2 * n] = -f_bot
        cnt[2 * n:3 * n] = ex
        cnt[3 * n:4 * n] = prod
        out[self.s_cnt] = cnt
        return out if vec else out[:, 0]

    # ------------------------------------------------------------- integrate
    def _solve(self, x0, t0, t1, kind, cv, acc, events=None):
        sol = solve_ivp(self.rhs, (t0, t1), x0, method="BDF", args=(kind, cv, acc), rtol=self.case.rtol,
                        atol=self.atol, vectorized=True, dense_output=True, events=events)
        if sol.status < 0:
            raise RuntimeError(f"{kind}: {sol.message}")
        return sol

    def run_step(self, x_start, step: Step):
        x0 = x_start.copy()
        x0[self.s_cnt] = 0.0
        segs, t0, prefill_t = [], 0.0, None
        if step.kind == "FEED":
            p_set = self.case.tank_p_set
            if (self.tank_pressure(x0) - p_set) / p_set < -1e-9:
                def ev(t, y, *args):
                    return self.tank_pressure(y) - p_set
                ev.terminal, ev.direction = True, 1
                sol = self._solve(x0, 0.0, step.duration, step.kind, step.cv, True, events=[ev])
                segs.append((0.0, sol.t[-1], sol.sol))
                x0 = sol.y[:, -1].copy()
                t0 = sol.t[-1]
                if sol.status == 1:
                    prefill_t = t0
        if step.duration - t0 > 1e-12 * step.duration:
            sol = self._solve(x0, t0, step.duration, step.kind, step.cv, False)
            segs.append((t0, step.duration, sol.sol))
            x_end = sol.y[:, -1].copy()
        else:
            x_end = x0
        return segs, x_end, prefill_t

    @staticmethod
    def sample(segs, t):
        t = np.atleast_1d(t)
        out = np.empty((segs[0][2](segs[0][0]).shape[0], t.size))
        for j, tj in enumerate(t):
            for (a, b, f) in segs:
                if tj <= b + 1e-12:
                    out[:, j] = f(min(max(tj, a), b))
                    break
            else:
                out[:, j] = segs[-1][2](segs[-1][1])
        return out

    def run_cycle(self, x_start):
        x = x_start.copy()
        recs = []
        n = self.n
        for s, step in enumerate(self.case.steps):
            inv0 = self.inventory(x)
            segs, x_end, prefill_t = self.run_step(x, step)
            cnt = x_end[self.s_cnt]
            feed_in, vent, to_tank, prod = cnt[0:n], cnt[n:2 * n], cnt[2 * n:3 * n], cnt[3 * n:4 * n]
            imbalance = feed_in - vent - prod - (self.inventory(x_end) - inv0)
            recs.append(dict(step=s, kind=step.kind, segs=segs, x0=x.copy(), x_end=x_end.copy(), feed_in=feed_in.copy(),
                             vent=vent.copy(), to_tank=to_tank.copy(), product=prod.copy(), imbalance=imbalance,
                             prefill_t=prefill_t))
            x = x_end.copy()
        return x, recs

    def css_change(self, x_new, x_old):
        comp = self.case.css_components
        dc = (x_new[self.s_c] - x_old[self.s_c]).reshape(self.n, self.N) / self.c_ref
        dq = (x_new[self.s_q] - x_old[self.s_q]).reshape(self.n, self.N) / self.q_ref
        dt = (x_new[self.s_t] - x_old[self.s_t]) / self.c_ref
        if comp is not None:
            dc, dq, dt = dc[list(comp)], dq[list(comp)], dt[list(comp)]
        d = np.concatenate([dc.ravel(), dq.ravel(), dt.ravel()])
        return float(np.sqrt(np.mean(d * d)))

    def cycle_metrics(self, recs):
        L, H = self.case.light, self.case.heavy
        feed = sum(r["feed_in"] for r in recs)
        vent = sum(r["vent"] for r in recs)
        prod = sum(r["product"] for r in recs)
        imb = np.array([r["imbalance"] for r in recs])
        with np.errstate(invalid="ignore", divide="ignore"):
            step_rel = np.where(feed > 0, np.abs(imb) / np.where(feed > 0, feed, 1.0), 0.0)
            css_rel = np.where(feed > 0, np.abs(feed - vent - prod) / np.where(feed > 0, feed, 1.0), 0.0)
        return dict(feed_mol=feed.tolist(), extract_mol=vent.tolist(), raffinate_mol=prod.tolist(),
                    purity_light=float(prod[L] / prod.sum()) if prod.sum() > 0 else float("nan"),
                    recovery_light=float(prod[L] / feed[L]),
                    purity_heavy_extract=float(vent[H] / vent.sum()), recovery_heavy=float(vent[H] / feed[H]),
                    step_balance_rel=float(step_rel.max()), css_balance_rel=float(css_rel.max()),
                    prefill_s=[r["prefill_t"] for r in recs if r["kind"] == "FEED"][0])

    def run_to_css(self, x0=None, progress=None, n_cycles=None):
        """Cycle to CSS (or exactly n_cycles cycles when given)."""
        x = self.initial_state() if x0 is None else x0.copy()
        hist, recs = [], None
        total = n_cycles or self.case.max_cycles
        for cyc in range(1, total + 1):
            x_new, recs = self.run_cycle(x)
            dx = self.css_change(x_new, x)
            m = self.cycle_metrics(recs)
            m.update(cycle=cyc, css_change=dx)
            hist.append(m)
            if progress:
                progress(m)
            x = x_new
            if n_cycles is None and dx < self.case.css_tol:
                break
        return x, recs, hist

    def profiles(self, recs, n_per_step):
        """Sampled cycle: t, y (n, N) per sample, P per sample (as psa-viz's ``profiles``, n components)."""
        if np.isscalar(n_per_step):
            n_per_step = [int(n_per_step)] * len(recs)
        T, Y, Q, P, PT = [], [], [], [], []
        t_off = 0.0
        for r, ns in zip(recs, n_per_step):
            step = self.case.steps[r["step"]]
            tt = np.linspace(0.0, step.duration, ns, endpoint=r is recs[-1])
            X = self.sample(r["segs"], tt)
            c, q, cR, _ = self.split(X)
            C = c.sum(0)
            T.append(t_off + tt)
            Y.append(np.moveaxis(c / C, -1, 0))
            Q.append(np.moveaxis(q, -1, 0))
            P.append(C.mean(0) * self.g.RT / 1e5)
            PT.append(cR.sum(0) * self.g.RT / 1e5)
            t_off += step.duration
        cat = np.concatenate
        return dict(t=cat(T), y=cat(Y), q=cat(Q), P=cat(P), p_tank=cat(PT), z=self.g.z / self.g.length)
