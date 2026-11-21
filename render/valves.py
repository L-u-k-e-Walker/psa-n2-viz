"""Valve drawings for the schematic (env viz; numpy + matplotlib). Picked with layout.valve_style.

Four styles, each a real valve type, drawn in design px on the full-frame axes (y down):
  ball      ball valves: a dark ball body with the bore cut out along the open path (an L bore in the
            3-way), so the tube and the moving gas show through it; a shut valve shows metal across
  butterfly butterfly valves: a round housing with the disc edge-on along the flow (open) or across it
            (shut); the 3-way is a flap diverter whose flap closes the unused branch; the purge needle
            is a disc set at an angle (a throttle)
  spool     spool valves (as in a valve block): the spool slides to cover the shut branch; the 2-way is
            a slide that retracts into its chamber
  piston    pneumatic angle-seat (piston) valves, pictorial: a body on the pipe and a 45-degree actuator
            cylinder, filled dark when the valve is open; the open passage shows as a light channel
Every 3-way valve has the bed on its common port and the two alternatives opposite each other
(inner line towards the middle of the plant, outer line towards the outside).
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path

VEC = {"u": (0.0, -1.0), "d": (0.0, 1.0), "l": (-1.0, 0.0), "r": (1.0, 0.0)}
STYLES = ("ball", "butterfly", "spool", "piston")


def _arc(x, y, r, a0, a1, n=24):
    a = np.linspace(a0, a1, n)
    return np.c_[x + r * np.cos(a), y + r * np.sin(a)]


def _perp(u):
    return np.array([-u[1], u[0]])


class ValveArt:
    def __init__(self, ax, T, scale, style, tube_w=16.0):
        assert style in STYLES, style
        self.ax, self.T, self.s, self.style = ax, T, scale, style
        self.hb = 0.5 * (tube_w - 3.0)  # bore half-width of the tubes (px)
        self.dark = T.get("valve_dark", T["ink2"])
        self.mid = T.get("valve_mid", T["metal"])
        self.light = T["tube_fill"]
        self.edge = T["tube_edge"]

    # ------------------------------------------------------------ primitives
    def lw(self, px):
        return px * 0.72 * self.s

    def _poly(self, pts, fc, ec="none", lw=0.0, z=6.0, **kw):
        self.ax.add_patch(Polygon(np.asarray(pts), closed=True, fc=fc, ec=ec, lw=self.lw(lw), zorder=z, **kw))

    def _circle(self, x, y, r, fc, ec="none", lw=0.0, z=6.0):
        self.ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=self.lw(lw), zorder=z))

    def _line(self, pts, color, lw, z=6.2, cap="round"):
        pts = np.asarray(pts)
        self.ax.plot(pts[:, 0], pts[:, 1], color=color, lw=self.lw(lw), zorder=z, solid_capstyle=cap)

    def _rrect(self, x0, y0, w, h, fc, ec="none", lw=0.0, rad=4.0, z=6.0):
        self.ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={rad:.1f}", fc=fc, ec=ec,
                                         lw=self.lw(lw), zorder=z))

    def _flanges(self, x, y, r, dirs):
        """A flange (short bar across the tube) where each port leaves the body."""
        for u in dirs:
            u = np.asarray(VEC[u] if isinstance(u, str) else u, float)
            c = np.array([x, y]) + u * (r + 2.5)
            p = _perp(u) * (self.hb + 3.5)
            self._line([c - p, c + p], self.edge, 2.6, z=5.8, cap="butt")

    # channel shapes (counter-clockwise in data coordinates), clipped to a circle of radius r
    def _band(self, x, y, r, u, h):
        th = math.atan2(u[1], u[0])
        a = math.asin(min(h / r, 0.99))
        return np.vstack([_arc(x, y, r, th - a, th + a, 10), _arc(x, y, r, th + math.pi - a, th + math.pi + a, 10)])

    def _ell(self, x, y, r, u1, u2, h):
        u1, u2 = np.asarray(u1, float), np.asarray(u2, float)
        if abs((math.atan2(u2[1], u2[0]) - math.atan2(u1[1], u1[0]) - math.pi / 2) % (2 * math.pi)) > 1e-6:
            u1, u2 = u2, u1  # u2 must be u1 turned by +90 degrees
        s_ = math.sqrt(r * r - h * h)
        a = math.asin(min(h / r, 0.99))
        t1, t2 = math.atan2(u1[1], u1[0]), math.atan2(u2[1], u2[0])
        c = np.array([x, y])

        def fr(p, q):
            return c + p * u1 + q * u2

        return np.vstack([fr(-h, -h)[None], fr(s_, -h)[None], _arc(x, y, r, t1 - a, t1 + a, 10), fr(h, h)[None],
                          fr(h, s_)[None], _arc(x, y, r, t2 - a, t2 + a, 10), fr(-h, s_)[None]])

    def _disk_with_hole(self, x, y, r, hole, fc, z=6.0):
        outer = _arc(x, y, r, 2 * math.pi, 0.0, 64)  # clockwise; the hole runs counter-clockwise
        verts = np.vstack([outer, outer[:1], hole, hole[:1]])
        codes = ([Path.MOVETO] + [Path.LINETO] * (len(outer) - 1) + [Path.CLOSEPOLY]
                 + [Path.MOVETO] + [Path.LINETO] * (len(hole) - 1) + [Path.CLOSEPOLY])
        self.ax.add_patch(PathPatch(Path(verts, codes), fc=fc, ec="none", zorder=z))

    # ------------------------------------------------------------- dispatch
    def two_way(self, x, y, open_, axis):
        """On/off valve on a straight pipe; axis 'h' or 'v'."""
        getattr(self, f"_{self.style}_2")(x, y, bool(open_), np.array((1.0, 0.0) if axis == "h" else (0.0, 1.0)))

    def three_way(self, x, y, common, inner, outer, open_side):
        """3-way valve: common port (the bed) to the inner or the outer line; open_side 'inner' | 'outer' | None."""
        getattr(self, f"_{self.style}_3")(x, y, common, inner, outer, open_side)

    def needle(self, x, y, axis):
        """Throttling (purge) valve, always open."""
        getattr(self, f"_{self.style}_n")(x, y, np.array((1.0, 0.0) if axis == "h" else (0.0, 1.0)))

    # ----------------------------------------------------------------- ball
    def _ball_2(self, x, y, open_, u):
        r = 15.5
        self._flanges(x, y, r, [u, -u])
        if open_:
            self._disk_with_hole(x, y, r, self._band(x, y, r - 0.01, u, self.hb), self.dark)
        else:
            self._circle(x, y, r, self.dark)
            self._poly(self._band(x, y, r - 2.2, _perp(u), self.hb - 0.5), self.mid, z=6.1)
        self._circle(x, y, r, "none", self.T["ink"], 1.2, z=6.3)

    def _ball_3(self, x, y, common, inner, outer, open_side):
        r = 17.5
        self._flanges(x, y, r, [common, inner, outer])
        if open_side:
            u2 = VEC[inner] if open_side == "inner" else VEC[outer]
            self._disk_with_hole(x, y, r, self._ell(x, y, r - 0.01, VEC[common], u2, self.hb), self.dark)
        else:
            self._circle(x, y, r, self.dark)
        self._circle(x, y, r, "none", self.T["ink"], 1.2, z=6.3)

    def _ball_n(self, x, y, u):
        r = 10.5
        self._flanges(x, y, r, [u, -u])
        self._disk_with_hole(x, y, r, self._band(x, y, r - 0.01, u, 2.4), self.dark)
        self._circle(x, y, r, "none", self.T["ink"], 1.2, z=6.3)
        p = _perp(u)
        c = np.array([x, y])
        self._line([c - p * r, c - p * (r + 8)], self.dark, 2.4)
        self._circle(*(c - p * (r + 11)), 3.6, self.bg_or_light(), self.dark, 1.6, z=6.4)

    def bg_or_light(self):
        return self.T["bg"]

    # ------------------------------------------------------------ butterfly
    def _housing(self, x, y, r):
        self._circle(x, y, r, self.light, self.dark, 2.2)

    def _pivot(self, x, y):
        self._circle(x, y, 2.6, self.T["bg"], self.dark, 1.3, z=6.5)

    def _butterfly_2(self, x, y, open_, u):
        r = 15.5
        self._flanges(x, y, r, [u, -u])
        self._housing(x, y, r)
        d = u if open_ else _perp(u)
        c = np.array([x, y])
        self._line([c - d * 0.78 * r, c + d * 0.78 * r], self.dark, 4.6, z=6.3)
        self._pivot(x, y)

    def _butterfly_3(self, x, y, common, inner, outer, open_side):
        r = 17.5
        self._flanges(x, y, r, [common, inner, outer])
        self._housing(x, y, r)
        c = np.array([x, y])
        if open_side:
            shut = np.asarray(VEC[outer] if open_side == "inner" else VEC[inner])
            m = c + shut * 0.52 * r
            p = _perp(shut) * 0.66 * r
            self._line([m - p, m + p], self.dark, 4.6, z=6.3)  # the flap across the shut branch
            self._line([c, m], self.dark, 1.8, z=6.25)
        self._pivot(x, y)

    def _butterfly_n(self, x, y, u):
        r = 11.0
        self._flanges(x, y, r, [u, -u])
        self._housing(x, y, r)
        a = math.radians(38.0)
        d = np.array([u[0] * math.cos(a) - u[1] * math.sin(a), u[0] * math.sin(a) + u[1] * math.cos(a)])
        c = np.array([x, y])
        self._line([c - d * 0.76 * r, c + d * 0.76 * r], self.dark, 3.6, z=6.3)
        self._pivot(x, y)

    # ---------------------------------------------------------------- spool
    # The housing is drawn UNDER the moving gas (z 2.8, between the tubes and the dots), so the dots are
    # seen passing through the open side of the valve; the spool (or slide) that closes the shut branch
    # and the outline are drawn over the gas.
    UNDER = 2.8

    def _spool_3(self, x, y, common, inner, outer, open_side):
        hl, hh = 27.0, 12.5  # half length along the inner-outer axis, half height
        if inner not in ("l", "r"):  # not used by the layouts (alternatives are horizontal)
            return self._ball_3(x, y, common, inner, outer, open_side)
        cvec = np.asarray(VEC[common])
        self._flanges(x, y, hl, [inner, outer])
        self._flanges(x, y, hh, [common])
        self._rrect(x - hl, y - hh, 2 * hl, 2 * hh, self.light, rad=5.0, z=self.UNDER)
        if open_side:
            shut = VEC[outer] if open_side == "inner" else VEC[inner]
            sg = shut[0]
            xa, xb = x + sg * 4.0, x + sg * (hl - 2.5)
            self._rrect(min(xa, xb), y - hh + 2.5, abs(xb - xa), 2 * hh - 5.0, self.dark, rad=3.0, z=6.2)
        self._rrect(x - hl, y - hh, 2 * hl, 2 * hh, "none", self.dark, 2.0, rad=5.0, z=6.3)
        # pilot on the side away from the bed
        pc = np.array([x, y]) - cvec * (hh + 4.5)
        self._rrect(pc[0] - 7.0, pc[1] - 3.5, 14.0, 7.0, self.mid, self.dark, 1.2, rad=2.0, z=6.1)

    def _spool_2(self, x, y, open_, u):
        p = _perp(u)
        side = -np.abs(p) if abs(p[0]) > 0.5 else np.array([0.0, -1.0])  # chamber to the left / above
        hw = self.hb + 3.0
        ch = 30.0  # chamber depth
        half_t = 10.0  # body thickness along the pipe, half
        c = np.array([x, y])
        self._flanges(x, y, half_t, [u, -u])
        a = side / np.linalg.norm(side)
        corners = [c - a * hw - u * half_t, c + a * (hw + ch) - u * half_t, c + a * (hw + ch) + u * half_t, c - a * hw + u * half_t]
        self._poly(corners, self.light, z=self.UNDER)
        sc = c + a * (hw + ch * 0.5 + 1.0) if open_ else c
        sh = hw - 0.5
        blk = [sc - a * sh - u * (half_t - 3), sc + a * sh - u * (half_t - 3), sc + a * sh + u * (half_t - 3), sc - a * sh + u * (half_t - 3)]
        self._poly(blk, self.dark, z=6.2)
        self._poly(corners, "none", self.dark, 2.0, z=6.3)
        rod0 = sc + a * sh
        self._line([rod0, c + a * (hw + ch + 6.0)], self.dark, 2.6, z=6.15, cap="butt")

    def _spool_n(self, x, y, u):
        p = _perp(u)
        a = -np.abs(p) if abs(p[0]) > 0.5 else np.array([0.0, -1.0])
        a = a / np.linalg.norm(a)
        c = np.array([x, y])
        hw, half_t = self.hb + 2.5, 8.0
        self._flanges(x, y, half_t, [u, -u])
        corners = [c - a * hw - u * half_t, c + a * (hw + 12) - u * half_t, c + a * (hw + 12) + u * half_t, c - a * hw + u * half_t]
        self._poly(corners, self.light, z=self.UNDER)
        tip = c - a * (self.hb * 0.25)
        self._poly([tip, c + a * hw - u * 4.0, c + a * hw + u * 4.0], self.dark, z=6.2)  # tapered needle into the bore
        self._poly(corners, "none", self.dark, 1.8, z=6.3)
        self._line([c + a * hw, c + a * (hw + 18)], self.dark, 2.4)
        self._circle(*(c + a * (hw + 21)), 3.6, self.T["bg"], self.dark, 1.6, z=6.4)

    # --------------------------------------------------------------- piston
    def _body(self, x, y, dirs, r=10.0):
        c = np.array([x, y])
        for u in dirs:
            u = np.asarray(VEC[u] if isinstance(u, str) else u, float)
            p = _perp(u) * (self.hb + 2.5)
            self._poly([c - p, c - p + u * (r + 5.0), c + p + u * (r + 5.0), c + p], self.dark, z=6.0)
        self._circle(x, y, r + 1.5, self.dark)

    def _passage(self, x, y, dirs, r=10.0):
        c = np.array([x, y])
        for u in dirs:
            u = np.asarray(VEC[u] if isinstance(u, str) else u, float)
            self._line([c, c + u * (r + 5.0)], self.light, 3.4, z=6.2)

    def _cylinder(self, x, y, d, open_, r=10.0, length=20.0, width=14.0):
        d = np.asarray(d, float) / np.linalg.norm(d)
        p = _perp(d)
        c = np.array([x, y])
        n0 = c + d * (r + 1.0)
        n1 = c + d * (r + 7.0)
        self._line([n0, n1], self.dark, 5.0, z=6.05, cap="butt")
        a0, a1 = n1, n1 + d * length
        q = p * width * 0.5
        self._poly([a0 - q, a1 - q, a1 + q, a0 + q], self.T["ink"] if open_ else self.T["bg"], self.T["ink"], 1.6, z=6.3)
        self._line([a1 - q * 0.75, a1 + q * 0.75], self.dark, 3.2, z=6.35, cap="butt")  # end cap
        mid = a0 + d * (length * 0.5)
        self._line([mid - q, mid + q], self.T["bg"] if open_ else self.edge, 1.0, z=6.36, cap="butt")

    def _piston_2(self, x, y, open_, u):
        self._body(x, y, [u, -u])
        if open_:
            self._passage(x, y, [u, -u])
        p = _perp(u)
        d = -np.abs(p) if abs(p[0]) > 0.5 else np.array([0.0, -1.0])  # actuator to the left / up
        self._cylinder(x, y, d, open_)

    def _piston_3(self, x, y, common, inner, outer, open_side):
        self._body(x, y, [common, inner, outer], r=11.0)
        if open_side:
            self._passage(x, y, [common, inner if open_side == "inner" else outer], r=11.0)
        d = -np.asarray(VEC[common]) + np.asarray(VEC[inner])  # 45 degrees, away from the bed, over the inner line
        self._cylinder(x, y, d, open_side is not None, r=11.0)

    def _piston_n(self, x, y, u):
        self._body(x, y, [u, -u], r=8.0)
        self._passage(x, y, [u, -u], r=8.0)
        p = _perp(u)
        a = -np.abs(p) if abs(p[0]) > 0.5 else np.array([0.0, -1.0])
        a = a / np.linalg.norm(a)
        c = np.array([x, y])
        self._line([c + a * 9.0, c + a * 20.0], self.dark, 3.0)
        self._line([c + a * 20.0 - _perp(a) * 7.0, c + a * 20.0 + _perp(a) * 7.0], self.dark, 3.2)  # hand wheel, edge-on

    # ------------------------------------------------------------ regulator
    def regulator(self, x, y, axis):
        """Pressure regulator: a body on the pipe and a spring bonnet with an adjusting screw (all styles)."""
        u = np.array((1.0, 0.0) if axis == "h" else (0.0, 1.0))
        self._body(x, y, [u, -u], r=8.0)
        p = _perp(u)
        a = np.abs(p)  # bonnet to the right / down
        a = a / np.linalg.norm(a)
        c = np.array([x, y])
        b0, b1 = c + a * 9.0, c + a * 31.0
        q = _perp(a) * 8.0
        self._poly([b0 - q, b1 - q, b1 + q, b0 + q], self.mid, self.dark, 1.4, z=6.2)
        zz = np.array([b0 + a * (2.5 + 2.6 * i) + q * (0.6 if i % 2 else -0.6) for i in range(8)])
        self._line(zz, self.dark, 1.1, z=6.3)
        self._line([b1, b1 + a * 6.0], self.dark, 2.4, z=6.25)
