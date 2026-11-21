"""Piping drawn as tubes with the gas moving through them (env viz; numpy + matplotlib).

Every pipe of the schematic is a polyline with
* a signed molar flow along its drawing direction, taken from the solver's valve flows of the frame
  (render/frames.py), and
* the composition of the gas in it: the upstream end of the flow (feed air, a bed's bottom or top
  gas, the receiver gas, or a mix of the beds' vent streams), also from the solver.
Dots move along the tube at v = vmax tanh(sqrt(|F| / F_ref)) (monotonic in the real flow, not to
scale, as the footnote says), so a closed line stands still and the equalization spikes saturate.
Each dot's species is drawn from the pipe's mole fractions with the same display weights as the bed
molecules (trace species exaggerated). Displacements are integrated over the 30 fps timeline and
rescaled so that every pipe returns to the same dot pattern at the loop seam.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Polygon

IN2, IO2, IAR, ICO2, IH2O = 0, 1, 2, 3, 4


def _mix(parts, fallback=None):
    """Flow-weighted mix of (flow >= 0, y (n,)) parts; ``fallback`` (the gas standing in the line) if
    nothing flows."""
    tot = sum(f for f, _ in parts)
    if tot <= 0:
        return fallback
    return sum(f * y for f, y in parts) / tot


class Pipe:
    def __init__(self, name, pts, flow, comp, tube_w=None, halo=False):
        """pts: [(x, y), ...] design px; flow(d) -> signed mol/s along pts (works on a frame dict or on
        the whole cache, element-wise); comp(fr, F) -> mole fractions (n,) of the gas in the pipe."""
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

    def point_at(self, s):
        """Positions (x, y) and unit tangents at arc lengths s (array)."""
        i = np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.seg_len) - 1)
        t = (s - self.cum[i]) / np.maximum(self.seg_len[i], 1e-9)
        p0, p1 = self.pts[i], self.pts[i + 1]
        tan = (p1 - p0) / np.maximum(self.seg_len[i], 1e-9)[:, None]
        return p0 + t[:, None] * (p1 - p0), tan


class PipeNet:
    def __init__(self, data, y_feed, weights, theme, scale, tube_w=12.0, spacing=13.0, vmax=260.0, f_ref=0.05,
                 fps=30.0):
        self.d = data
        self.yf = np.asarray(y_feed, float)
        self.w = np.asarray(weights, float)
        self.T = theme
        self.s = scale
        self.tube_w, self.spacing, self.vmax, self.f_ref, self.fps = tube_w, spacing, vmax, f_ref, fps
        self.pipes: list[Pipe] = []
        self.disp = {}

    # ------------------------------------------------------------ building
    def add(self, name, pts, flow, comp, tube_w=None, halo=False):
        self.pipes.append(Pipe(name, pts, flow, comp, tube_w=tube_w or self.tube_w, halo=halo))

    def speed(self, F):
        return self.vmax * np.sign(F) * np.tanh(np.sqrt(np.abs(F) / self.f_ref))

    def finish(self):
        """Integrate each pipe's dot displacement over the timeline (loop-seamless)."""
        n = len(self.d["t_video"])
        dt = 1.0 / self.fps
        for p in self.pipes:
            v = self.speed(np.asarray(p.flow(self.d), float) * np.ones(n))
            s = np.concatenate([[0.0], np.cumsum(v[:-1] * dt)])
            total = float(np.sum(v) * dt)
            target = round(total / self.spacing) * self.spacing
            if abs(total) > 1e-9:
                s *= target / total
            self.disp[p.name] = s

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
        f = int(round(float(fr["t_video"]) * self.fps)) % len(self.d["t_video"])
        dot = 6.0  # px diameter (design)
        for p in self.pipes:
            half = max(0.0, 0.5 * (p.tube_w - 3.0) - 0.5 * dot - 0.5)
            F = float(p.flow(fr))
            y = p.comp(fr, F)
            if y is None:
                continue
            wy = np.maximum(np.asarray(y, float), 0) * self.w
            if wy.sum() <= 0:
                continue
            cdf = np.cumsum(wy) / wy.sum()
            s0 = self.disp[p.name][f]
            nd = int(math.ceil(p.L / self.spacing)) + 1
            j = np.arange(nd)
            raw = j * self.spacing + s0
            wraps = np.floor(raw / (nd * self.spacing)).astype(int)
            pos = raw - wraps * nd * self.spacing
            ok = pos <= p.L
            j, wraps, pos = j[ok], wraps[ok], pos[ok]
            h = (j * 73856093 ^ (wraps + 1000) * 19349663) % 1000003 / 1000003.0  # stable per dot and pass
            h2 = (j * 83492791 ^ (wraps + 1000) * 2654435761) % 1000003 / 1000003.0
            sp = np.searchsorted(cdf, h)
            xy, tan = p.point_at(pos)
            nrm = np.c_[-tan[:, 1], tan[:, 0]]
            xy = xy + nrm * ((h2 - 0.5) * 2 * half)[:, None]
            alpha = 0.95 if abs(F) > 1e-5 else 0.45  # gas standing in a closed line drawn dimmer
            ax.scatter(xy[:, 0], xy[:, 1], s=(dot * self.s * 0.72) ** 2, c=[colors[i] for i in np.clip(sp, 0, len(colors) - 1)],
                       linewidths=0, alpha=alpha, zorder=z)

    def draw_chevrons(self, ax, fr, z=3.5):
        """A small direction mark on each flowing pipe (for stills)."""
        T = self.T
        for p in self.pipes:
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
