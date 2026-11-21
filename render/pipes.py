"""Piping drawn as tubes with the gas moving through them (env viz; numpy + matplotlib).

Every pipe of the schematic is a polyline with
* a signed molar flow F along its drawing direction (the solver's valve flows of the frame),
* the pressure P of the gas in it (the bed it opens to, the feed header, the receiver or the vent),
* the composition of the gas entering it: the upstream end of the flow (feed air, a bed's bottom or
  top gas, the receiver gas, or a mix of the beds' vent streams), also from the solver.
The dots are particles, simulated over the 30 fps timeline (one warm-up loop, then the stored loop):
* number density proportional to P^gamma (gamma = 1 by default: the gas's molar density, as for the
  bed molecules), dot_spacing_px apart at the feed pressure;
* they move with the actual volumetric flow, v = vmax tanh(sqrt((|F|/P) / (F_ref/P_ref))) px per
  video second (monotonic, not to scale): expanding exhaust gas runs fast and sparse, compressed
  feed slow and dense; a closed line stands still;
* new dots enter at the upstream end at the rate the flow brings them, and appear or vanish along
  the line when its pressure rises or falls;
* each dot keeps the species it was born with (drawn from the entering gas's mole fractions with
  the bed molecules' display weights, trace species exaggerated), so composition changes travel
  along the pipes with the flow.
"""
from __future__ import annotations

import hashlib

import numpy as np
from matplotlib.patches import Polygon

IN2, IO2, IAR, ICO2, IH2O = 0, 1, 2, 3, 4
DOT = 6.0  # px diameter (design)


def _mix(parts, fallback=None):
    """Flow-weighted mix of (flow >= 0, y (n,)) parts; ``fallback`` (the gas standing in the line) if
    nothing flows."""
    tot = sum(f for f, _ in parts)
    if tot <= 0:
        return fallback
    return sum(f * y for f, y in parts) / tot


def _insert_in_gaps(pos, k, L, rng):
    """k new positions, each in the middle of the currently largest gap (ends included): even spacing."""
    out = []
    cur = np.sort(pos)
    for _ in range(k):
        edges = np.concatenate([[0.0], cur, [L]])
        g = np.diff(edges)
        i = int(np.argmax(g))
        x = edges[i] + g[i] * (0.5 + 0.12 * (rng.random() - 0.5))
        out.append(x)
        cur = np.insert(cur, i, x)
    return np.array(out)


def _remove_from_crowds(pos, k):
    """Indices of k dots to remove, each the one with the smallest gap to its neighbours."""
    idx = np.argsort(pos)
    cur = list(idx)
    drop = []
    for _ in range(min(k, len(cur))):
        x = pos[cur]
        if len(cur) == 1:
            j = 0
        else:
            gl = np.diff(np.concatenate([[x[0] - 1e9], x]))
            gr = np.diff(np.concatenate([x, [x[-1] + 1e9]]))
            j = int(np.argmin(np.minimum(gl, gr)))
        drop.append(cur.pop(j))
    return np.array(drop, int)


class Pipe:
    def __init__(self, name, pts, flow, comp, tube_w=None, halo=False, pressure=None, gas=True):
        """pts: [(x, y), ...] design px; flow(d) -> signed mol/s along pts and pressure(d) -> bar abs (both
        work on a frame dict or element-wise on the whole cache); comp(fr, F) -> mole fractions (n,)."""
        self.name = name
        self.pts = np.asarray(pts, float)
        seg = np.diff(self.pts, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.L = float(self.cum[-1])
        self.flow = flow
        self.comp = comp
        self.tube_w = tube_w
        self.halo = halo
        self.pressure = pressure
        self.gas = gas

    def point_at(self, s):
        """Positions (x, y) and unit tangents at arc lengths s (array)."""
        s = np.asarray(s, float)
        i = np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.seg_len) - 1)
        t = (s - self.cum[i]) / np.maximum(self.seg_len[i], 1e-9)
        p0, p1 = self.pts[i], self.pts[i + 1]
        tan = (p1 - p0) / np.maximum(self.seg_len[i], 1e-9)[:, None]
        return p0 + t[:, None] * (p1 - p0), tan


class PipeNet:
    def __init__(self, data, y_feed, weights, theme, scale, tube_w=12.0, spacing=13.0, vmax=260.0, f_ref=0.05,
                 fps=30.0, p_ref=None, density_exponent=1.0):
        self.d = data
        self.yf = np.asarray(y_feed, float)
        self.w = np.asarray(weights, float)
        self.T = theme
        self.s = scale
        self.tube_w, self.spacing, self.vmax, self.f_ref, self.fps = tube_w, spacing, vmax, f_ref, fps
        self.p_ref = p_ref if p_ref is not None else 8.5
        self.gamma = density_exponent
        self.pipes: list[Pipe] = []
        self.state = {}

    # ------------------------------------------------------------ building
    def add(self, name, pts, flow, comp, tube_w=None, halo=False, pressure=None, gas=True):
        self.pipes.append(Pipe(name, pts, flow, comp, tube_w=tube_w or self.tube_w, halo=halo, pressure=pressure, gas=gas))

    def speed(self, F, P):
        q = (np.abs(F) / np.maximum(P, 0.05)) / (self.f_ref / self.p_ref)
        return self.vmax * np.sign(F) * np.tanh(np.sqrt(q))

    def _frames(self):
        d = self.d
        n = len(d["t_video"])
        return [{k: (v[f] if (isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n) else v) for k, v in d.items()}
                for f in range(n)]

    def _cdf(self, y):
        if y is None:
            return None
        wy = np.maximum(np.asarray(y, float), 0.0) * self.w
        tot = wy.sum()
        return None if tot <= 0 else np.cumsum(wy) / tot

    def finish(self):
        """Simulate every gas-carrying pipe's dots over the timeline (warm-up loop, then the stored loop)."""
        d = self.d
        n = len(d["t_video"])
        dt = 1.0 / self.fps
        frames = self._frames()
        for p in self.pipes:
            if not p.gas:
                continue
            F = np.asarray(p.flow(d), float) * np.ones(n)
            P = (np.asarray(p.pressure(d), float) if p.pressure is not None else np.full(n, self.p_ref)) * np.ones(n)
            P = np.maximum(P, 0.05)
            v = self.speed(F, P)
            dens = (P / self.p_ref) ** self.gamma / self.spacing  # dots per px
            seed = int(hashlib.sha256(p.name.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            half = max(0.0, 0.5 * (p.tube_w - 3.0) - 0.5 * DOT - 0.5)
            cdf0 = self._cdf(p.comp(frames[0], F[0]))
            n0 = int(round(dens[0] * p.L)) if cdf0 is not None else 0
            pos = (np.arange(n0) + 0.5 + 0.3 * (rng.random(n0) - 0.5)) * (p.L / max(n0, 1))  # even, jittered
            sp = np.searchsorted(cdf0, rng.random(n0)) if n0 else np.zeros(0, int)
            lat = (rng.random(n0) - 0.5) * 2 * half
            carry = 0.0
            out_pos, out_sp, out_lat, off = [], [], [], [0]
            for step in range(2 * n):
                f = step % n
                pos = pos + v[f] * dt
                keep = (pos >= 0.0) & (pos <= p.L)
                pos, sp, lat = pos[keep], sp[keep], lat[keep]
                target = dens[f] * p.L
                cdf = self._cdf(p.comp(frames[f], F[f]))
                if cdf is not None and abs(v[f]) > 0.5:  # inflow at the upstream end
                    carry += dens[f] * abs(v[f]) * dt
                    k = int(carry)
                    carry -= k
                    k = min(k, max(0, int(round(target - len(pos)))) + 1)
                    if k > 0:
                        u = (np.arange(k) + rng.random(k)) / k * abs(v[f]) * dt
                        pos = np.concatenate([pos, u if v[f] > 0 else p.L - u])
                        sp = np.concatenate([sp, np.searchsorted(cdf, rng.random(k))])
                        lat = np.concatenate([lat, (rng.random(k) - 0.5) * 2 * half])
                tol = max(1.0, 0.08 * target)
                diff = target - len(pos)
                if diff > tol and cdf is not None:  # compression: the line fills (new dots in the largest gaps)
                    k = int(round(diff - 0.5 * tol))
                    pos = np.concatenate([pos, _insert_in_gaps(pos, k, p.L, rng)])
                    sp = np.concatenate([sp, np.searchsorted(cdf, rng.random(k))])
                    lat = np.concatenate([lat, (rng.random(k) - 0.5) * 2 * half])
                elif diff < -tol:  # expansion: the line empties (dots leave the most crowded spots)
                    k = min(len(pos), int(round(-diff - 0.5 * tol)))
                    keep = np.ones(len(pos), bool)
                    keep[_remove_from_crowds(pos, k)] = False
                    pos, sp, lat = pos[keep], sp[keep], lat[keep]
                if step >= n:
                    out_pos.append(pos.astype(np.float32))
                    out_sp.append(np.clip(sp, 0, len(self.w) - 1).astype(np.int8))
                    out_lat.append(lat.astype(np.float32))
                    off.append(off[-1] + len(pos))
            self.state[p.name] = (np.concatenate(out_pos), np.concatenate(out_sp), np.concatenate(out_lat), np.array(off))

    # -------------------------------------------------------------- drawing
    def draw_tubes(self, ax, z=2):
        T = self.T
        edge, fill = T["tube_edge"], T["tube_fill"]
        for n, p in enumerate(self.pipes):
            kw = dict(solid_joinstyle="round", solid_capstyle="round")
            zz = z + 0.001 * n  # later pipes over earlier ones (branches over headers)
            tw = p.tube_w * self.s * 0.72  # px -> pt at 100 dpi
            if p.halo:
                ax.plot(p.pts[:, 0], p.pts[:, 1], color=T["bg"], lw=tw + 6 * self.s * 0.72, zorder=zz, solid_capstyle="butt")
            ax.plot(p.pts[:, 0], p.pts[:, 1], color=edge, lw=tw, zorder=zz + 0.0002, **kw)
            ax.plot(p.pts[:, 0], p.pts[:, 1], color=fill, lw=tw - 3.0 * self.s * 0.72, zorder=zz + 0.0004, **kw)
            ax.plot(p.pts[:, 0], p.pts[:, 1], color=T["tube_shine"], lw=0.22 * tw, alpha=0.5, zorder=zz + 0.0006, **kw)

    def draw_gas(self, ax, fr, colors, z=3.0):
        """Dots of frame fr (frame index from its video time)."""
        n = len(self.d["t_video"])
        f = int(round(float(fr["t_video"]) * self.fps)) % n
        for p in self.pipes:
            if p.name not in self.state:
                continue
            pos, sp, lat, off = self.state[p.name]
            a, b = off[f], off[f + 1]
            if b <= a:
                continue
            xy, tan = p.point_at(pos[a:b])
            nrm = np.c_[-tan[:, 1], tan[:, 0]]
            xy = xy + nrm * lat[a:b, None]
            F = float(p.flow(fr))
            alpha = 0.95 if abs(F) > 1e-5 else 0.45  # gas standing in a closed line drawn dimmer
            ax.scatter(xy[:, 0], xy[:, 1], s=(DOT * self.s * 0.72) ** 2, c=[colors[i] for i in sp[a:b]],
                       linewidths=0, alpha=alpha, zorder=z)

    def draw_chevrons(self, ax, fr, z=3.5):
        """A small direction mark on each flowing pipe (for stills)."""
        T = self.T
        for p in self.pipes:
            if not p.gas:
                continue
            F = float(p.flow(fr))
            if abs(F) < 1e-4 or p.L < 40:
                continue
            xy, tan = p.point_at(np.array([0.5 * p.L]))
            t = tan[0] * (1 if F > 0 else -1)
            nrm = np.array([-t[1], t[0]])
            c = xy[0]
            a = 0.62 * p.tube_w
            tri = [c + t * a, c - t * a * 0.6 + nrm * a * 0.75, c - t * a * 0.6 - nrm * a * 0.75]
            ax.add_patch(Polygon(tri, closed=True, fc=T["ink"], ec=T["bg"], lw=0.8 * self.s, alpha=0.9, zorder=z))
