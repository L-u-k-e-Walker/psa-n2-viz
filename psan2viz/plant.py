"""Two-bed PSA plant: both beds, the product receiver and the valves as one ODE system.

Each bed is a ``BedGrid`` (layered: alumina at the feed end, CMS above) using the shared
kernel from ``psan2viz.bed`` (psa-viz's equations, generalized). Valves are linear,
molar flow = Cv (P_upstream - P_downstream), and bidirectional:

  bottom of each bed: feed (header -> bed), vent (bed -> atmosphere through a silencer),
                      bottom equalization (bed <-> other bed)
  top of each bed:    product (bed <-> receiver; flow back into the bed early in the
                      production step is the product back-flow), purge (receiver ->
                      bed, needle valve), top equalization (bed <-> other bed)

The receiver is well mixed; product leaves it through an outlet regulator at a constant
molar flow (scaled down linearly only if the receiver falls within 0.2 bar of the
delivery pressure, which the gates check never happens at CSS).

Cycle (literature cycle, docs/MISSION.md): four segments,
  EQ (t_eq):    bed 1 -> bed 0 through top and bottom equalization valves
  HALF (t_prod): bed 0 feed + product valves open (pressurization with back-flow, then
                production); bed 1 vent + purge open (blowdown with simultaneous purge)
  EQ, HALF:     the same with the beds swapped.
Because the second half mirrors the first, the cyclic steady state is the fixed point of
the half-cycle map followed by a bed swap; ``run_to_css`` iterates that map with
Anderson acceleration and the gates verify the result with plain full cycles.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .bed import BedGrid

VALVES_BOTTOM = ("feed", "vent", "eq_bot")
VALVES_TOP = ("product", "purge", "eq_top")
COUNTERS = ("feed0", "feed1", "vent0", "vent1", "ptank0", "ptank1", "purge0", "purge1", "eqt", "eqb", "product")


@dataclass
class Valves:
    feed: float  # mol/(s bar)
    vent: float
    product: float
    purge: float
    eq_top: float
    eq_bot: float


@dataclass
class Segment:
    name: str  # "EQ" or "HALF"
    duration: float  # s
    open0: frozenset  # valves open on bed 0
    open1: frozenset  # valves open on bed 1


@dataclass
class PlantCase:
    grid: BedGrid
    species: tuple
    y_feed: np.ndarray  # (n,) feed (wet air)
    p_feed: float  # bar abs, header
    p_vent: float  # bar abs
    valves: Valves
    t_eq: float  # s
    t_prod: float  # s
    tank_volume: float  # m^3
    product_flow: float  # mol/s through the outlet regulator
    p_deliver: float  # bar abs, regulator outlet
    y_vent_in: np.ndarray | None = None  # composition if air ever enters through the vent (never at CSS)
    init_p_low: float = 1.2
    init_p_high: float = 7.5
    init_y: np.ndarray | None = None  # initial bed and receiver gas
    rtol: float = 1e-6
    atol_rel: float = 1e-9
    atol_rel_trace: float | None = None  # looser absolute tolerance for CO2 and H2O (None = atol_rel)
    provenance: dict = field(default_factory=dict)

    @property
    def t_half(self):
        return self.t_eq + self.t_prod

    @property
    def t_cycle(self):
        return 2.0 * self.t_half

    def segments(self):
        eq = frozenset({"eq_top", "eq_bot"})
        prod, regen = frozenset({"feed", "product"}), frozenset({"vent", "purge"})
        return [Segment("EQ", self.t_eq, eq, eq), Segment("HALF", self.t_prod, prod, regen),
                Segment("EQ", self.t_eq, eq, eq), Segment("HALF", self.t_prod, regen, prod)]


class Plant:
    def __init__(self, case: PlantCase):
        self.case = case
        g = self.g = case.grid
        n, N = self.n, self.N = g.n, g.N
        self.RT = g.RT
        self.yf = np.asarray(case.y_feed, float)
        self.yv = self.yf if case.y_vent_in is None else np.asarray(case.y_vent_in, float)
        nb = 2 * n * N  # one bed: c then q
        self.s_bed = [slice(0, nb), slice(nb, 2 * nb)]
        self.s_t = slice(2 * nb, 2 * nb + n)
        self.s_cnt = slice(2 * nb + n, 2 * nb + n + len(COUNTERS) * n)
        self.size = self.s_cnt.stop
        self.n_phys = self.s_cnt.start  # state entries that carry physics (not counters)
        # per-species scales for tolerances and the CSS norm
        C_f = case.p_feed * 1e5 / self.RT
        self.c_scale = C_f * np.maximum(self.yf, 1e-4)  # (n,)
        p_f = self.yf * case.p_feed
        qf = g.equilibrium((self.yf * C_f)[:, None, None] * np.ones((1, N, 1)))[:, :, 0]  # (n, N)
        self.q_scale = np.maximum(qf, 1e-6 * max(qf.max(), 1e-12))
        self.feed_mol_scale = np.maximum(self.yf, 1e-4) * C_f * g.volume  # rough moles per bed volume
        bed_scale = np.concatenate([np.repeat(self.c_scale, N), self.q_scale.ravel()])
        scale = np.concatenate([bed_scale, bed_scale, self.c_scale, np.tile(self.feed_mol_scale, len(COUNTERS))])
        self.scale = scale
        rel = np.full(self.size, case.atol_rel)
        if case.atol_rel_trace is not None:  # CO2 and H2O: resolved to atol_rel_trace of their feed level
            sp = list(case.species)
            tr = np.zeros(n, bool)
            tr[[sp.index("CO2"), sp.index("H2O")]] = True
            bed_tr = np.concatenate([np.repeat(tr, N), np.repeat(tr, N)])
            mask = np.concatenate([bed_tr, bed_tr, tr, np.tile(tr, len(COUNTERS))])
            rel[mask] = case.atol_rel_trace
        self.atol = rel * scale
        self.n_rhs = 0
        del p_f

    # ------------------------------------------------------------------ views
    def bed(self, x, k):
        n, N = self.n, self.N
        tail = x.shape[1:]
        v = x[self.s_bed[k]]
        return v[:n * N].reshape((n, N) + tail), v[n * N:].reshape((n, N) + tail)

    def tank(self, x):
        return x[self.s_t]

    def counters(self, x):
        return x[self.s_cnt].reshape((len(COUNTERS), self.n) + x.shape[1:])

    def bed_pressure(self, x, k):
        c, _ = self.bed(x, k)
        return c.sum(0).mean(0) * self.RT / 1e5

    def tank_pressure(self, x):
        return x[self.s_t].sum(0) * self.RT / 1e5

    def inventory(self, x):
        """(beds (2, n), tank (n,)) moles."""
        beds = np.array([self.g.inventory(*self.bed(x, k)) for k in (0, 1)])
        return beds, self.case.tank_volume * x[self.s_t]

    # --------------------------------------------------------------- the ODE
    def _valve_flows(self, X, seg):
        """Per-valve flows INTO each bed: dict[(k, valve)] = (F (K,), Ncomp (n, K)); plus helpers."""
        cs = self.case
        v = cs.valves
        RT = self.RT
        K = X.shape[1]
        beds = [self.bed(X, k) for k in (0, 1)]
        C = [b[0].sum(0) for b in beds]
        Pb = [Ck[0] * RT / 1e5 for Ck in C]
        Pt = [Ck[-1] * RT / 1e5 for Ck in C]
        yb = [beds[k][0][:, 0] / C[k][0] for k in (0, 1)]
        yt = [beds[k][0][:, -1] / C[k][-1] for k in (0, 1)]
        cT = X[self.s_t]
        CT = cT.sum(0)
        PT = CT * RT / 1e5
        yT = cT / CT
        yf = self.yf[:, None]
        yv = self.yv[:, None]
        flows = {}
        for k in (0, 1):
            j = 1 - k
            opened = seg.open0 if k == 0 else seg.open1
            for name in opened:
                if name == "feed":
                    F, ys, ye = v.feed * (cs.p_feed - Pb[k]), yf, yb[k]
                elif name == "vent":
                    F, ys, ye = v.vent * (cs.p_vent - Pb[k]), yv, yb[k]
                elif name == "eq_bot":
                    F, ys, ye = v.eq_bot * (Pb[j] - Pb[k]), yb[j], yb[k]
                elif name == "product":
                    F, ys, ye = v.product * (PT - Pt[k]), yT, yt[k]
                elif name == "purge":
                    F, ys, ye = v.purge * (PT - Pt[k]), yT, yt[k]
                elif name == "eq_top":
                    F, ys, ye = v.eq_top * (Pt[j] - Pt[k]), yt[j], yt[k]
                else:
                    raise ValueError(name)
                flows[(k, name)] = (F, F * np.where(F >= 0.0, ys, ye))
        zero = (np.zeros(K), np.zeros((self.n, K)))
        return flows, beds, C, PT, yT, zero

    def product_withdrawal(self, PT):
        cs = self.case
        return cs.product_flow * np.clip((PT - cs.p_deliver) / 0.2, 0.0, 1.0)

    def rhs(self, t, x, seg):
        self.n_rhs += 1
        vec = x.ndim == 2
        X = x if vec else x[:, None]
        K = X.shape[1]
        n, N = self.n, self.N
        flows, beds, C, PT, yT, zero = self._valve_flows(X, seg)
        out = np.empty_like(X)
        for k in (0, 1):
            bot = [flows[(k, v)] for v in VALVES_BOTTOM if (k, v) in flows]
            top = [flows[(k, v)] for v in VALVES_TOP if (k, v) in flows]
            Fb = sum(f[0] for f in bot) if bot else zero[0]
            Nb = sum(f[1] for f in bot) if bot else zero[1]
            Ft = sum(f[0] for f in top) if top else zero[0]
            Nt = sum(f[1] for f in top) if top else zero[1]
            c, q = beds[k]
            dc, dq, _, _, _ = self.g.rates(c, q, C[k], ("valves", Fb, Nb), ("valves", Ft, Nt))
            sl = self.s_bed[k]
            out[sl.start:sl.start + n * N] = dc.reshape(n * N, K)
            out[sl.start + n * N:sl.stop] = dq.reshape(n * N, K)
        get = lambda k, v: flows.get((k, v), zero)[1]  # noqa: E731
        FP = self.product_withdrawal(PT)
        to_tank = -(get(0, "product") + get(1, "product") + get(0, "purge") + get(1, "purge"))
        out[self.s_t] = (to_tank - FP * yT) / self.case.tank_volume
        cnt = np.stack([get(0, "feed"), get(1, "feed"), -get(0, "vent"), -get(1, "vent"),
                        -get(0, "product"), -get(1, "product"), get(0, "purge"), get(1, "purge"),
                        -get(0, "eq_top"), -get(0, "eq_bot"), FP * yT])
        out[self.s_cnt] = cnt.reshape(len(COUNTERS) * n, K)
        return out if vec else out[:, 0]

    # ------------------------------------------------------------- integrate
    def run_segment(self, x_start, seg):
        x0 = x_start.copy()
        x0[self.s_cnt] = 0.0
        sol = solve_ivp(self.rhs, (0.0, seg.duration), x0, method="BDF", args=(seg,), rtol=self.case.rtol,
                        atol=self.atol, vectorized=True, dense_output=True)
        if sol.status < 0:
            raise RuntimeError(f"{seg.name}: {sol.message}")
        return sol

    def initial_state(self):
        """Bed 0 low (just regenerated), bed 1 high (just produced); N2-rich gas, alumina dry."""
        cs = self.case
        x = np.zeros(self.size)
        y0 = np.asarray(cs.init_y if cs.init_y is not None else self.default_init_y(), float)
        for k, p in ((0, cs.init_p_low), (1, cs.init_p_high)):
            c, q = self.bed(x, k)
            c[:] = (y0 * p * 1e5 / self.RT)[:, None]
            q[:] = self.g.equilibrium(c[:, :, None])[:, :, 0]
        x[self.s_t] = y0 * cs.init_p_high * 1e5 / self.RT
        return x

    def default_init_y(self):
        sp = list(self.case.species)
        y = np.zeros(self.n)
        y[sp.index("N2")], y[sp.index("O2")], y[sp.index("Ar")] = 0.985, 0.004, 0.011
        return y / y.sum()

    def swap(self, x):
        y = x.copy()
        y[self.s_bed[0]], y[self.s_bed[1]] = x[self.s_bed[1]], x[self.s_bed[0]]
        return y

    def run_segments(self, x_start, segs):
        x = x_start.copy()
        recs = []
        for i, seg in enumerate(segs):
            inv0 = self.inventory(x)
            sol = self.run_segment(x, seg)
            x_end = sol.y[:, -1].copy()
            inv1 = self.inventory(x_end)
            cnt = self.counters(x_end)
            d = {name: cnt[j].copy() for j, name in enumerate(COUNTERS)}
            feed = d["feed0"] + d["feed1"]
            vent = d["vent0"] + d["vent1"]
            sys_imb = feed - vent - d["product"] - ((inv1[0].sum(0) + inv1[1]) - (inv0[0].sum(0) + inv0[1]))
            bed_imb = [d[f"feed{k}"] - d[f"vent{k}"] - d[f"ptank{k}"] + d[f"purge{k}"]
                       + (-1 if k == 0 else 1) * (d["eqt"] + d["eqb"]) - (inv1[0][k] - inv0[0][k]) for k in (0, 1)]
            tank_imb = d["ptank0"] + d["ptank1"] - d["purge0"] - d["purge1"] - d["product"] - (inv1[1] - inv0[1])
            recs.append(dict(seg=i, name=seg.name, duration=seg.duration, open0=sorted(seg.open0),
                             open1=sorted(seg.open1), sol=sol, x0=x.copy(), x_end=x_end, counters=d,
                             sys_imbalance=sys_imb, bed_imbalance=bed_imb, tank_imbalance=tank_imb,
                             n_steps=len(sol.t)))
            x = x_end
        return x, recs

    def run_cycle(self, x_start):
        return self.run_segments(x_start, self.case.segments())

    def half_map(self, x):
        """Half cycle (EQ + HALF with bed 0 producing) followed by the bed swap."""
        x_end, recs = self.run_segments(x, self.case.segments()[:2])
        return self.swap(x_end), recs

    # ------------------------------------------------------------------- CSS
    def norm_change(self, x_new, x_old):
        d = (x_new[:self.n_phys] - x_old[:self.n_phys]) / self.scale[:self.n_phys]
        return float(np.sqrt(np.mean(d * d))), float(np.max(np.abs(d)))

    def project(self, x):
        """Keep a (possibly extrapolated) state physical: no negative amounts."""
        y = x.copy()
        y[:self.n_phys] = np.maximum(y[:self.n_phys], 0.0)
        return y

    def run_to_css(self, x0=None, tol=1e-8, max_iter=400, m=8, beta=1.0, plain=6, progress=None):
        """Fixed point of the swapped half-cycle map, Anderson-accelerated (type II).

        ``plain`` initial plain iterations (to leave the start-up transient), then Anderson
        with history ``m``. Returns (x_css, history)."""
        x = self.initial_state() if x0 is None else x0.copy()
        P = self.n_phys
        s = self.scale[:P]
        hist, dX, dF = [], [], []
        x_prev = f_prev = None
        t0 = time.time()
        for it in range(1, max_iter + 1):
            gx, _ = self.half_map(x)
            f = (gx[:P] - x[:P]) / s
            rms = float(np.sqrt(np.mean(f * f)))
            hist.append(dict(iter=it, rms=rms, max=float(np.max(np.abs(f))), wall_s=time.time() - t0))
            if progress:
                progress(hist[-1])
            if rms < tol:
                x = gx
                break
            z = x[:P] / s
            if it <= plain or m == 0:
                z_new = z + f
            else:
                if x_prev is not None:
                    dX.append(z - x_prev)
                    dF.append(f - f_prev)
                    if len(dX) > m:
                        dX.pop(0)
                        dF.pop(0)
                if dF:
                    Fm = np.array(dF).T
                    gamma, *_ = np.linalg.lstsq(Fm, f, rcond=None)
                    z_new = z + beta * f - (np.array(dX).T + beta * Fm) @ gamma
                else:
                    z_new = z + f
                if len(hist) > 2 and rms > 10 * min(h["rms"] for h in hist[:-1]):
                    dX.clear()
                    dF.clear()  # restart the history after a bad extrapolation
                    z_new = z + f
            x_prev, f_prev = z, f
            x_new = x.copy()
            x_new[:P] = z_new * s
            x = self.project(x_new)
        return x, hist

    def run_to_css_nk(self, x0=None, tol=2e-7, plain=12, newton_max=20, krylov=40, eta=1e-2, h_rms=1e-4,
                      progress=None):
        """CSS by Jacobian-free Newton-Krylov on the swapped half-cycle map (normalized state).

        ``plain`` plain iterations first (start-up transient), then Newton steps: GMRES solves
        (G' - I) d = -(G(z) - z) with finite-difference Jacobian-vector products (one half-cycle
        each, perturbation of RMS size ``h_rms``), then a backtracking line search on the residual
        RMS with the state kept physical. Returns (x_css, history)."""
        from scipy.sparse.linalg import LinearOperator, gmres

        P = self.n_phys
        s = self.scale[:P]
        hist = []
        t0 = time.time()
        n_maps = [0]

        def G(x):
            n_maps[0] += 1
            gx, _ = self.half_map(x)
            return gx

        def note(kind, rms, mx, extra=""):
            hist.append(dict(iter=len(hist) + 1, kind=kind, rms=rms, max=mx, maps=n_maps[0], wall_s=time.time() - t0))
            if progress:
                progress(dict(hist[-1], note=extra))

        x = self.initial_state() if x0 is None else x0.copy()
        gx = G(x)
        for _ in range(plain):
            f = (gx[:P] - x[:P]) / s
            rms = float(np.sqrt(np.mean(f * f)))
            note("plain", rms, float(np.abs(f).max()))
            if rms < tol:
                return gx, hist
            x = self.project(gx)
            gx = G(x)
        for _ in range(newton_max):
            f = (gx[:P] - x[:P]) / s
            rms = float(np.sqrt(np.mean(f * f)))
            note("newton", rms, float(np.abs(f).max()))
            if rms < tol:
                return gx, hist
            z = x[:P] / s

            def Jv(v, x=x, z=z, f=f):
                nv = float(np.linalg.norm(v))
                if nv == 0.0:
                    return np.zeros_like(v)
                h = h_rms * np.sqrt(P) / nv
                xp = x.copy()
                xp[:P] = (z + h * v) * s
                fp = (G(xp)[:P] - xp[:P]) / s
                return (fp - f) / h

            A = LinearOperator((P, P), matvec=Jv, dtype=float)
            d, info = gmres(A, -f, rtol=eta, restart=krylov, maxiter=2)
            accepted, rn, fn, lam = False, rms, f, 0.0
            for lam in (1.0, 0.5, 0.25, 0.1):
                xn = x.copy()
                xn[:P] = (z + lam * d) * s
                xn = self.project(xn)
                gn = G(xn)
                fn = (gn[:P] - xn[:P]) / s
                rn = float(np.sqrt(np.mean(fn * fn)))
                if rn < rms:
                    x, gx, accepted = xn, gn, True
                    break
            if progress:
                progress(dict(iter=len(hist), kind="step", rms=rn, max=float(np.abs(fn).max()), maps=n_maps[0],
                              wall_s=time.time() - t0, note=f"gmres info {info}, lambda {lam}, accepted {accepted}"))
            if not accepted:  # fall back to a plain iteration
                x = self.project(gx)
                gx = G(x)
        return gx, hist

    def slow_sets(self):
        """Groups of slowly relaxing state variables, each extrapolated with its own decay factor:
        the CO2 and H2O profiles (gas + loading) in both beds and the receiver. (The bulk loadings
        decay at 0.5-0.7 per cycle; extrapolating them as well set up a period-two oscillation.)"""
        sp = list(self.case.species)
        n, N = self.n, self.N

        def rows(i, part):
            off = 0 if part == "c" else n * N
            return np.concatenate([np.arange(sl.start + off + i * N, sl.start + off + (i + 1) * N) for sl in self.s_bed])

        out = {}
        for name in ("CO2", "H2O"):
            i = sp.index(name)
            out[name] = np.concatenate([rows(i, "q"), rows(i, "c")])
        out["receiver"] = np.arange(self.s_t.start, self.s_t.stop)
        return out

    def run_to_css_proj(self, x0=None, tol=2e-7, plain=12, max_maps=3000, m_max=3000.0, cos_min=0.95,
                        relax=2, progress=None, checkpoint=None):
        """CSS by plain half-cycle maps with vector-Aitken extrapolation of the slow variables.

        The CO2 and H2O profiles (trace species: they barely act back on the bulk flow) relax over
        hundreds of half-cycles; the receiver and the bulk loadings relax over tens. Changes are measured over full cycles (two swapped half-maps),
        because the bed-to-bed difference flips sign at every swap and hides the slow drift in
        consecutive half-map changes. After ``relax`` plain maps following a jump (the fast response),
        five maps give, for each group (``slow_sets``), two consecutive
        full-cycle changes d1, d2 of its variables give a decay factor lam = d2.d1 / d1.d1 per cycle;
        if they point the same way (cosine >= cos_min) and 0 < lam < 1, the loadings jump by the
        rest of the geometric series, lam/(1-lam) d2 (capped at m_max cycles). All other variables
        relax by the plain maps in between. Returns (x_css, history)."""
        P = self.n_phys
        s = self.scale[:P]
        sets = self.slow_sets()
        hist = []
        t0 = time.time()
        n_maps = [0]

        def G(x):
            n_maps[0] += 1
            gx, _ = self.half_map(x)
            return gx

        def resid(x, gx):
            f = (gx[:P] - x[:P]) / s
            return float(np.sqrt(np.mean(f * f))), float(np.abs(f).max())

        x = self.initial_state() if x0 is None else x0.copy()
        for _ in range(plain):
            gx = G(x)
            rms, mx = resid(x, gx)
            hist.append(dict(iter=len(hist) + 1, kind="plain", rms=rms, max=mx, maps=n_maps[0], wall_s=time.time() - t0))
            if progress:
                progress(dict(hist[-1], note=""))
            if rms < tol:
                return gx, hist
            x = self.project(gx)
        jumped = False
        beta, rms_prev = 1.0, None  # jump damping: halves after a step whose residual rose, recovers by 1.25x
        while n_maps[0] < max_maps:
            if jumped:
                for _ in range(relax):
                    x = self.project(G(x))
            x1 = G(x)
            x2 = G(x1)
            x3 = G(x2)
            x4 = G(x3)
            x5 = G(x4)
            rms, mx = resid(x4, x5)
            notes = []
            if rms < tol:
                hist.append(dict(iter=len(hist) + 1, kind="done", rms=rms, max=mx, maps=n_maps[0], wall_s=time.time() - t0))
                if progress:
                    progress(dict(hist[-1], note=""))
                return x5, hist
            if rms_prev is not None:
                beta = max(0.25, 0.5 * beta) if rms > rms_prev else min(1.0, 1.25 * beta)
            rms_prev = rms
            xn = x5.copy()
            for name, idx in sets.items():
                d1 = (x3[idx] - x1[idx]) / s[idx]
                d2 = (x5[idx] - x3[idx]) / s[idx]
                n1, n2 = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
                if n1 == 0.0 or n2 == 0.0:
                    continue
                cos = float(d1 @ d2) / (n1 * n2)
                lam = float(d2 @ d1) / (n1 * n1)
                if cos >= cos_min and 0.0 < lam < 1.0:
                    m = beta * min(lam / (1.0 - lam), m_max)
                    xn[idx] = x5[idx] + m * d2 * s[idx]
                    notes.append(f"{name}: lam {lam:.4f} jump {m:.0f}")
                else:
                    notes.append(f"{name}: no jump (cos {cos:.2f}, lam {lam:.3f})")
            jumped = any("jump" in nt and "no jump" not in nt for nt in notes)
            notes.append(f"damping {beta:.2f}")
            x = self.project(xn)
            hist.append(dict(iter=len(hist) + 1, kind="aitken", rms=rms, max=mx, maps=n_maps[0], wall_s=time.time() - t0))
            if progress:
                progress(dict(hist[-1], note="; ".join(notes)))
            if checkpoint:
                checkpoint(x)
        return x, hist

    def run_to_css_aa(self, x0=None, tol=2e-7, m=10, max_cycles=400, progress=None, checkpoint=None):
        """CSS by Anderson acceleration of the full-cycle map on the slow variables.

        Each iteration runs one full cycle (two swapped half-cycle maps, so the beds return to their
        own order and the bed-to-bed sign flip of the half-map does not appear). The slow variables
        (the CO2 and H2O profiles, the receiver, the bulk loadings) are updated jointly by type-II
        Anderson mixing over the last ``m`` iterations, in per-species normalized units; all other
        variables take the plain map's value. The history restarts if the residual grows 3x.
        Converged when the RMS of the normalized full-cycle change < tol. Returns (x_css, history)."""
        P = self.n_phys
        s = self.scale[:P]
        sp = list(self.case.species)
        n, N = self.n, self.N

        def rows(i, part):
            off = 0 if part == "c" else n * N
            return np.concatenate([np.arange(sl.start + off + i * N, sl.start + off + (i + 1) * N) for sl in self.s_bed])

        S = np.concatenate([rows(sp.index("CO2"), "c"), rows(sp.index("CO2"), "q"), rows(sp.index("H2O"), "c"),
                            rows(sp.index("H2O"), "q"), np.arange(self.s_t.start, self.s_t.stop)]
                           + [rows(sp.index(k), "q") for k in ("N2", "O2", "Ar")])
        sS = s[S]
        x = self.initial_state() if x0 is None else x0.copy()
        hist, dX, dF = [], [], []
        z_prev = f_prev = None
        best = np.inf
        t0 = time.time()
        for it in range(1, max_cycles + 1):
            y = self.half_map(self.half_map(x)[0])[0]  # one full cycle (two swaps = original order)
            full = (y[:P] - x[:P]) / s
            rms = float(np.sqrt(np.mean(full * full)))
            hist.append(dict(iter=it, kind="anderson", rms=rms, max=float(np.abs(full).max()), maps=2 * it,
                             wall_s=time.time() - t0))
            if progress:
                progress(dict(hist[-1], note=f"history {len(dF)}"))
            if rms < tol:
                return y, hist
            z = x[S] / sS
            f = (y[S] - x[S]) / sS
            if rms > 3 * best:
                dX.clear()
                dF.clear()
                z_prev = f_prev = None
            best = min(best, rms)
            if z_prev is not None:
                dX.append(z - z_prev)
                dF.append(f - f_prev)
                if len(dX) > m:
                    dX.pop(0)
                    dF.pop(0)
            z_prev, f_prev = z, f
            xn = y.copy()
            if dF:
                Fm = np.array(dF).T
                gamma, *_ = np.linalg.lstsq(Fm, f, rcond=None)
                z_new = z + f - (np.array(dX).T + Fm) @ gamma
                xn[S] = z_new * sS
            x = self.project(xn)
            if checkpoint:
                checkpoint(x)
        return x, hist

    def trace_indices(self):
        """State indices of the trace species (CO2, H2O): gas and loading in both beds, and the receiver."""
        sp = list(self.case.species)
        n, N = self.n, self.N
        idx = []
        for name in ("CO2", "H2O"):
            i = sp.index(name)
            for sl in self.s_bed:
                idx.append(np.arange(sl.start + i * N, sl.start + (i + 1) * N))
                idx.append(np.arange(sl.start + n * N + i * N, sl.start + n * N + (i + 1) * N))
            idx.append(np.array([self.s_t.start + i]))
        return np.concatenate(idx)

    def full_cycle(self, x):
        """One full cycle as two swapped half-cycle maps (the beds end in their own order)."""
        return self.half_map(self.half_map(x)[0])[0]

    def run_to_css_trace_nk(self, x0, tol=2e-7, newton_max=10, krylov=30, gmres_restarts=2, eta=1e-2, h_rms=1e-4,
                            settle=2, progress=None, checkpoint=None):
        """CSS of the slow trace profiles by Jacobian-free Newton-Krylov on the full-cycle map.

        The trace species (CO2, H2O) barely act back on the bulk flow, and their profiles relax over
        hundreds to thousands of cycles (the water front in the alumina translates slowly, CO2 moves
        with it). After ``settle`` plain full cycles (bulk at its CSS), Newton steps act on the trace
        subspace T only: F(z) = (G(x(z))[T] - x[T]) / s[T] with x(z) the current state with its trace
        part set to z (normalized), G the full-cycle map. GMRES solves J d = -F with finite-difference
        products (one full cycle each), then a backtracking line search on |F|; the accepted map output
        becomes the next state (bulk and trace advance one cycle). Converged when the RMS of the
        normalized full-cycle change of the whole state < tol. Returns (x_css, history)."""
        from scipy.sparse.linalg import LinearOperator, gmres

        P = self.n_phys
        s = self.scale[:P]
        T = self.trace_indices()
        sT = s[T]
        hist = []
        t0 = time.time()
        n_cyc = [0]

        def G(x):
            n_cyc[0] += 1
            return self.full_cycle(x)

        def full_rms(x, y):
            f = (y[:P] - x[:P]) / s
            return float(np.sqrt(np.mean(f * f)))

        def note(kind, rms, rmsT, extra=""):
            hist.append(dict(iter=len(hist) + 1, kind=kind, rms=rms, rms_trace=rmsT, maps=2 * n_cyc[0],
                             wall_s=time.time() - t0))
            if progress:
                progress(dict(hist[-1], note=extra))

        x = x0.copy()
        y = G(x)
        for _ in range(settle):
            x = self.project(y)
            y = G(x)
        for _ in range(newton_max):
            rms = full_rms(x, y)
            F = (y[T] - x[T]) / sT
            rmsT = float(np.sqrt(np.mean(F * F)))
            note("newton", rms, rmsT)
            if rms < tol:
                return y, hist
            z = x[T] / sT

            def Jv(v, x=x, z=z, F=F):
                nv = float(np.linalg.norm(v))
                if nv == 0.0:
                    return np.zeros_like(v)
                h = h_rms * np.sqrt(T.size) / nv
                xp = x.copy()
                xp[T] = (z + h * v) * sT
                yp = G(xp)
                return ((yp[T] - xp[T]) / sT - F) / h

            A = LinearOperator((T.size, T.size), matvec=Jv, dtype=float)
            d, info = gmres(A, -F, rtol=eta, restart=krylov, maxiter=gmres_restarts)
            accepted = False
            for lam in (1.0, 0.5, 0.25, 0.1):
                xn = x.copy()
                xn[T] = (z + lam * d) * sT
                xn = self.project(xn)
                yn = G(xn)
                Fn = (yn[T] - xn[T]) / sT
                if float(np.sqrt(np.mean(Fn * Fn))) < rmsT:
                    x, y, accepted = xn, yn, True
                    break
            note("step", full_rms(x, y), float(np.sqrt(np.mean(((y[T] - x[T]) / sT) ** 2))),
                 f"gmres info {info}, lambda {lam}, accepted {accepted}")
            if not accepted:
                x = self.project(y)
                y = G(x)
            if checkpoint:
                checkpoint(x)
        return y, hist

    def residual_breakdown(self, x, gx):
        """RMS of the normalized half-map residual per (bed part, species): diagnostics for CSS solvers."""
        n, N = self.n, self.N
        f = (gx - x) / self.scale
        out = {}
        sp = list(self.case.species)
        for k, sl in enumerate(self.s_bed):
            blk = f[sl]
            for part, off in (("c", 0), ("q", n * N)):
                for i, name in enumerate(sp):
                    v = blk[off + i * N: off + (i + 1) * N]
                    out[f"bed{k}.{part}.{name}"] = float(np.sqrt(np.mean(v * v)))
        out["tank"] = float(np.sqrt(np.mean(f[self.s_t] ** 2)))
        return out

    # --------------------------------------------------------------- results
    def cycle_metrics(self, recs):
        cs = self.case
        sp = list(cs.species)
        iN2, iO2, iAr, iCO2, iH2O = (sp.index(s) for s in ("N2", "O2", "Ar", "CO2", "H2O"))
        tot = {key: sum(r["counters"][key] for r in recs) for key in COUNTERS}
        feed = tot["feed0"] + tot["feed1"]
        vent = tot["vent0"] + tot["vent1"]
        prod = tot["product"]
        purge = tot["purge0"] + tot["purge1"]
        y_p = prod / prod.sum()
        t_cyc = sum(r["duration"] for r in recs)
        imb = np.array([r["sys_imbalance"] for r in recs])
        bed_imb = np.array([np.max(np.abs(b)) for r in recs for b in r["bed_imbalance"]])
        feed_cycle = np.maximum(feed, 1e-300)
        v_cms = 2 * self.g.area * sum(lay.height for lay in self.g.layers if "CMS" in lay.name.upper())
        return dict(
            feed_mol=feed.tolist(), vent_mol=vent.tolist(), product_mol=prod.tolist(), purge_mol=purge.tolist(),
            product_O2_ppm=float(1e6 * y_p[iO2]), product_Ar_pct=float(100 * y_p[iAr]),
            product_CO2_ppm=float(1e6 * y_p[iCO2]), product_H2O_ppm=float(1e6 * y_p[iH2O]),
            purity_N2_plus_Ar_pct=float(100 * (1 - y_p[iO2] - y_p[iCO2] - y_p[iH2O])),
            purity_industry_pct=float(100 * (1 - y_p[iO2])),
            N2_excl_Ar_pct=float(100 * y_p[iN2]),
            recovery_N2=float(prod[iN2] / feed[iN2]),
            recovery_N2_Ar=float((prod[iN2] + prod[iAr]) / (feed[iN2] + feed[iAr])),
            air_factor=float(feed.sum() / prod.sum()),
            product_mol_s=float(prod.sum() / t_cyc),
            productivity_Nm3_h_m3=float(prod.sum() / t_cyc * 0.022413969 * 3600 / v_cms) if v_cms > 0 else None,
            purge_to_product=float(purge.sum() / prod.sum()),
            step_balance_rel=float(np.max(np.abs(imb) / feed_cycle[None, :])),
            css_balance_rel=float(np.max(np.abs(feed - vent - prod) / feed_cycle)),
            bed_balance_max_mol=float(bed_imb.max()),
            t_cycle_s=t_cyc,
        )
