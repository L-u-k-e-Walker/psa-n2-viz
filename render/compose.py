"""Frame composition (matplotlib): beds, P&ID overlay, panels, legend (env viz).

Everything drawn comes from one frame of the physics cache (render/frames.py) and the cycle
metrics; nothing is scripted. Coordinates are design pixels on a 1920x1080 canvas (y from the
top), scaled to the tier's size. The beds are drawn flat from the solver profiles in the mock
tier; the draft and final tiers paste the OVITO render into the same bed boxes.
Origin: the panel style, direct labels and loading ramps follow psa-viz column/render/render_column.py.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipes import PipeNet, _mix  # noqa: E402
from psan2viz.style import SPECIES, SPECIES_ROLE, SPECIES_TEX, THEMES, apply_ramp, loading_ramp, mpl_rc  # noqa: E402

W, H = 1920, 1080
IN2, IO2, IAR, ICO2, IH2O = 0, 1, 2, 3, 4  # solver order
BAR_ATM = 1.01325

# ----------------------------------------------------------------------------- layouts
# Boxes are (x0, y0, x1, y1) in design px, y from the top.
# prof: "pairs" = O2 + trace strips beside each bed; "shared" = one O2 + one trace panel for both beds
LAYOUTS = {
    "A": dict(  # plant on the left two thirds, profile strips beside each bed, dashboard column on the right
        name="plant + dashboard column", eq_x=735, title=(36, 24), legend=(36, 128),
        beds=[(330, 330, 450, 815), (800, 330, 920, 815)], prof="pairs", label_side=["left", "left"],
        prof_boxes=[[(468, 330, 563, 815), (575, 330, 670, 815)], [(938, 330, 1033, 815), (1045, 330, 1140, 815)]],
        feed_y=965, vent_y=1008, top_y=272, prod_y=212, train_x=40, receiver=(1155, 150, 1215, 290), out="down",
        timing=(1330, 52, 1890, 318), pressure=(1330, 382, 1890, 532), o2=(1330, 600, 1890, 726),
        readouts=(1330, 770), footnote=(36, 1070)),
    "B": dict(  # beds close together with one shared profile block between them, dashboard on the right
        name="beds with shared profiles", eq_x=620, title=(36, 24), legend=(1330, 866),
        beds=[(185, 360, 305, 760), (900, 360, 1020, 760)], prof="shared", label_side=["left", "right"],
        prof_boxes=[(380, 360, 600, 760), (640, 360, 860, 760)],
        eq_bot_drop=72,  # bottom equalization pipe below the profile x-axis labels
        bot_split=44, top_split=44, purge_gap=26,
        train="compact",
        feed_y=975, vent_y=1020, top_y=300, prod_y=150, train_x=20, receiver=(1170, 104, 1230, 244), out="down",
        timing=(1330, 52, 1890, 300), pressure=(1330, 352, 1890, 488), o2=(1330, 540, 1890, 656),
        readouts=(1330, 696), footnote=(36, 1072)),
    "C": dict(  # wide plant on top with profile strips, a strip of time panels along the bottom
        name="plant on top, time strip below", eq_x=985, title=(36, 24), legend=(36, 300),
        beds=[(560, 215, 660, 612), (1110, 215, 1210, 612)], prof="pairs", label_side=["left", "left"],
        prof_boxes=[[(678, 215, 763, 612), (775, 215, 860, 612)], [(1228, 215, 1313, 612), (1325, 215, 1410, 612)]],
        feed_y=715, vent_y=758, top_y=165, prod_y=112, train_x=300, receiver=(1500, 70, 1560, 205), out="right",
        timing=(150, 830, 800, 1005), pressure=(880, 830, 1310, 1005), o2=(1390, 830, 1880, 1005),
        readouts=(1500, 300), footnote=(36, 1072)),
}

STEP_NAMES = {"EQ_up": "equalization (up)", "EQ_down": "equalization (down)", "press": "pressurization\n+ back-flow",
              "prod": "production", "regen": "blowdown\n+ purge"}


def bed_step(fr, k):
    """Display name of bed k's step from the segment and the product-valve flow direction."""
    if str(fr["seg_name"]) == "EQ":
        return "EQ_up" if fr[f"P{k}"] < fr[f"P{1 - k}"] else "EQ_down"
    if fr[f"open{k}_feed"]:
        return "press" if fr[f"F{k}_product"] > 0 else "prod"
    return "regen"


def spread(values, min_gap):
    """Shift label positions apart (keeping order) so neighbours are at least min_gap apart."""
    order = np.argsort(values)
    out = np.asarray(values, float).copy()
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


class Composer:
    def __init__(self, theme, layout, data, metrics, cycle, size=(W, H), cfg=None):
        self.theme, self.T = theme, THEMES[theme]
        self.L = LAYOUTS[layout]
        self.layout = layout
        self.d = data
        self.met = metrics
        self.cyc = cycle
        self.size = size
        self.s = size[0] / W
        self.cfg = cfg or {}
        self.unit = self.cfg.get("layout", {}).get("pressure_unit", "barg")
        self.ramp_cms = loading_ramp(theme, "O2", deep=True)
        self.ramp_al = loading_ramp(theme, "H2O", deep=True, bare_key="bare_alumina")
        g = cycle.pl.g
        self.cms_cells = g.layer_of == [i for i, lay in enumerate(g.layers) if lay.name == "CMS"][0]
        self.q_ref = cycle.pl.q_scale
        self.zf = data["z_faces"]
        self.z = data["z"]
        self.z_al = float(data["z_al"])
        self.t_cycle = cycle.T
        self.starts = cycle.starts / cycle.T
        fl = [k for k in data.keys() if k.startswith("F0_") or k.startswith("F1_")]
        # arrow scale: the feed flow during production (98th percentile of the feed valves); the short
        # equalization and pressurization spikes saturate at full width instead of shrinking every other arrow
        feed = np.abs(np.concatenate([np.asarray(data["F0_feed"]), np.asarray(data["F1_feed"])]))
        self.f_ref = float(np.percentile(feed[feed > 0], 98)) if np.any(feed > 0) else max(float(np.max(np.abs(data[k]))) for k in fl)
        self._backflow = self._backflow_windows()
        self.nozzles = None  # per bed (y_top, y_bottom) in design px where the 3D nozzles end (draft/final)
        o2max = 100 * max(float(np.max(data["y0"][:, IO2])), float(np.max(data["y1"][:, IO2])))
        self.o2_xmax = 10 * math.ceil(o2max / 10 + 0.05)

    # ---------------------------------------------------------------- helpers
    def fx(self, x):
        return x / W

    def fy(self, y):
        return 1 - y / H

    def fs(self, pt):
        return pt * self.s

    def box_axes(self, fig, box, **kw):
        x0, y0, x1, y1 = box
        return fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H], **kw)

    def text(self, fig, x, y, s, size=14, color=None, **kw):
        kw.setdefault("va", "top")
        return fig.text(self.fx(x), self.fy(y), s, fontsize=self.fs(size), color=color or self.T["ink"], **kw)

    def p_disp(self, p_bar):
        return p_bar - BAR_ATM if self.unit == "barg" else p_bar

    def _backflow_windows(self):
        out = {0: [], 1: []}
        d = self.d
        frac = d["t_cycle"] / self.t_cycle
        for k in (0, 1):
            m = np.asarray(d[f"open{k}_feed"], bool) & (d[f"F{k}_product"] > 0)
            if m.any():
                idx = np.nonzero(m)[0]
                for s in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                    out[k].append((float(frac[s[0]]), float(frac[min(s[-1] + 1, len(frac) - 1)])))
        return out

    # ------------------------------------------------------------------ frame
    def draw(self, fr, out, beds_3d=None, watermark=None):
        T = self.T
        with plt.rc_context(mpl_rc(self.theme, self.s)):
            fig = plt.figure(figsize=(self.size[0] / 100, self.size[1] / 100), dpi=100)
            self._pid(fig, fr)
            if beds_3d is None:
                self._beds_flat(fig, fr)
            else:  # [(RGBA image, (x, y) of its top-left corner in output px), ...]
                Wp, Hp = self.size
                for img, (ox, oy) in beds_3d:
                    h, w = img.shape[:2]
                    ax = fig.add_axes([ox / Wp, 1 - (oy + h) / Hp, w / Wp, h / Hp], zorder=1)
                    ax.imshow(img, interpolation="nearest")
                    ax.set_axis_off()
            self._bed_labels(fig, fr)
            self._profiles(fig, fr)
            self._timing(fig, fr)
            self._pressure(fig, fr)
            self._o2(fig, fr)
            self._readouts(fig, fr)
            self._legend(fig)
            self._title(fig, fr)
            self.text(fig, self.L["footnote"][0], self.L["footnote"][1], self.cfg.get("text", {}).get("footnote", ""),
                      size=10.5, color=T["muted"], va="bottom")
            if watermark:
                fig.text(0.985, 0.012, watermark, ha="right", va="bottom", fontsize=self.fs(26), color=T["muted"],
                         fontweight="bold", alpha=0.85)
            fig.savefig(out, dpi=100)
            plt.close(fig)

    # ------------------------------------------------------------------- beds
    def bed_geometry(self, k):
        return self.L["beds"][k]

    def _beds_flat(self, fig, fr):
        T = self.T
        ax = fig.add_axes([0, 0, 1, 1], zorder=1)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_axis_off()
        for k in (0, 1):
            x0, y0, x1, y1 = self.bed_geometry(k)
            hpx = y1 - y0
            q = fr[f"q{k}"]
            v = np.where(self.cms_cells, q[IO2] / self.q_ref[IO2], q[IH2O] / self.q_ref[IH2O])
            cols = np.where(self.cms_cells[:, None], apply_ramp(self.ramp_cms, v), apply_ramp(self.ramp_al, v))
            for n in range(len(v)):
                ya, yb = y1 - self.zf[n] * hpx, y1 - self.zf[n + 1] * hpx
                ax.add_patch(Rectangle((x0, yb), x1 - x0, ya - yb + 0.6, color=cols[n], lw=0))
            ax.add_patch(FancyBboxPatch((x0 - 6, y0 - 14), x1 - x0 + 12, hpx + 28, boxstyle="round,pad=0,rounding_size=14",
                                        fc="none", ec=T["metal"], lw=2.5 * self.s))
            ya = y1 - self.z_al * hpx
            ax.plot([x0, x1], [ya, ya], color=T["bg"], lw=1.0 * self.s, alpha=0.6)

    def _bed_labels(self, fig, fr):
        T = self.T
        for k in (0, 1):
            x0, y0, x1, y1 = self.bed_geometry(k)
            left = self.L.get("label_side", ["left", "left"])[k] == "left"
            xl, ha = (x0 - 16, "right") if left else (x1 + 16, "left")
            self.text(fig, xl, y0 + 4, f"Bed {k + 1}", size=16, ha=ha, fontweight="bold")
            self.text(fig, xl, y0 + 30, STEP_NAMES[bed_step(fr, k)], size=12, ha=ha, color=T["ink2"], linespacing=1.25)
            self.text(fig, xl, y1 - self.z_al * (y1 - y0) * 0.5, "alumina", size=10, ha=ha, va="center", color=T["muted"])
            self.text(fig, xl, y1 - 0.55 * (y1 - y0), "CMS", size=10, ha=ha, va="center", color=T["muted"])

    # -------------------------------------------------------------------- P&ID
    def _valve(self, ax, x, y, open_, vertical=True, size=13, needle=False):
        T = self.T
        s = size
        if vertical:
            tri = ([(x - s * 0.8, y - s), (x + s * 0.8, y - s), (x, y)], [(x - s * 0.8, y + s), (x + s * 0.8, y + s), (x, y)])
        else:
            tri = ([(x - s, y - s * 0.8), (x - s, y + s * 0.8), (x, y)], [(x + s, y - s * 0.8), (x + s, y + s * 0.8), (x, y)])
        fc = T["ink"] if open_ else T["bg"]
        ec = T["ink"] if open_ else T["muted"]
        for t in tri:
            ax.add_patch(Polygon(t, closed=True, fc=fc, ec=ec, lw=1.6 * self.s, zorder=6))
        if needle:
            ax.plot([x - s * 1.3, x + s * 1.3], [y + s * 1.3, y - s * 1.3], color=ec, lw=1.2 * self.s, zorder=7)

    def _arrow(self, ax, p0, p1, flow, color):
        """Arrow along a pipe from p0 to p1 (positive flow) or back (negative); width ~ sqrt(|flow|)."""
        a = abs(flow) / self.f_ref
        if a < 2e-3:
            return
        w = 2.0 + 9.0 * math.sqrt(min(a, 1.0))
        if flow < 0:
            p0, p1 = p1, p0
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=f"-|>,head_width={1.1 * w * self.s:.2f},head_length={1.3 * w * self.s:.2f}",
                                     mutation_scale=1, lw=w * self.s, color=color, alpha=0.35 + 0.65 * min(1, 4 * a),
                                     zorder=8, shrinkA=0, shrinkB=0))

    def _net(self):
        """The pipe network for the current nozzle geometry (built once per geometry)."""
        key = None if self.nozzles is None else tuple(map(tuple, self.nozzles))
        if getattr(self, "_net_key", "unset") == key:
            return self._net_obj, self._geo
        L, T = self.L, self.T
        beds = [self.bed_geometry(k) for k in (0, 1)]
        cx = [0.5 * (b[0] + b[2]) for b in beds]
        dx = 26
        fy, vy, ty, py = L["feed_y"], L["vent_y"], L["top_y"], L["prod_y"]
        pq = py + L.get("purge_gap", 24)
        if self.nozzles is not None:
            y_tops = [nz[0] for nz in self.nozzles]
            y_bot = max(nz[1] for nz in self.nozzles)
        else:
            y_tops = [b[1] - 14 for b in beds]
            y_bot = beds[0][3] + 14
        node_b = y_bot + L.get("eq_bot_drop", 30)
        split_b = node_b + L.get("bot_split", 26)
        node_t = ty
        split_t = node_t - L.get("top_split", 20)
        mid = L.get("eq_x", 0.5 * (cx[0] + cx[1]))
        tx = L["train_x"]
        compact = L.get("train") == "compact"
        x_in = tx + (170 if compact else 300)
        vx = max(cx[1] + dx, beds[1][2]) + 90
        rx0, ry0, rx1, ry1 = L["receiver"]
        rcx = 0.5 * (rx0 + rx1)
        yf = self.cyc.pl.yf
        mcfg = self.cfg.get("molecules", {}).get("weights", {})
        w = [mcfg.get(sp, 1.0) for sp in SPECIES]
        pcfg = self.cfg.get("pipes", {})
        net = PipeNet(self.d, yf, w, T, self.s, tube_w=pcfg.get("tube_px", 12.0), spacing=pcfg.get("dot_spacing_px", 13.0),
                      vmax=pcfg.get("vmax_px_s", 260.0), f_ref=pcfg.get("f_ref_mol_s", 0.05))
        bot = lambda fr, k: fr[f"y{k}"][:, 0]  # noqa: E731
        top = lambda fr, k: fr[f"y{k}"][:, -1]  # noqa: E731
        tank = lambda fr: fr["y_tank"]  # noqa: E731

        def pos(x):
            return np.maximum(x, 0.0)

        # --- headers first (branches cross over them with a halo)
        net.add("feed_hdr_a", [(x_in, fy), (cx[0] - dx, fy)], lambda d: d["F0_feed"] + d["F1_feed"], lambda fr, F: yf)
        net.add("feed_hdr_b", [(cx[0] - dx, fy), (cx[1] - dx, fy)], lambda d: d["F1_feed"], lambda fr, F: yf)
        net.add("vent_hdr_a", [(cx[0] + dx, vy), (cx[1] + dx, vy)], lambda d: -d["F0_vent"], lambda fr, F: bot(fr, 0))
        net.add("vent_hdr_b", [(cx[1] + dx, vy), (vx, vy)], lambda d: -(d["F0_vent"] + d["F1_vent"]),
                lambda fr, F: _mix([(pos(-fr["F0_vent"]), bot(fr, 0)), (pos(-fr["F1_vent"]), bot(fr, 1))]) if F > 0 else bot(fr, 0))
        net.add("prod_hdr_a", [(cx[0] - dx, py), (cx[1] - dx, py)], lambda d: -d["F0_product"],
                lambda fr, F: top(fr, 0) if F > 0 else tank(fr))
        net.add("prod_hdr_b", [(cx[1] - dx, py), (rx0, py)], lambda d: -(d["F0_product"] + d["F1_product"]),
                lambda fr, F: _mix([(pos(-fr["F0_product"]), top(fr, 0)), (pos(-fr["F1_product"]), top(fr, 1))]) if F > 0 else tank(fr))
        net.add("eq_bot", [(cx[0], node_b), (cx[1], node_b)], lambda d: d["F1_eq_bot"],
                lambda fr, F: bot(fr, 0) if F > 0 else bot(fr, 1))
        net.add("eq_top", [(cx[0], node_t), (cx[1], node_t)], lambda d: d["F1_eq_top"],
                lambda fr, F: top(fr, 0) if F > 0 else top(fr, 1))
        net.add("product_out", [(rcx, ry1), (rcx, ry1 + 96)], lambda d: d["F_product_out"], lambda fr, F: tank(fr))
        # feed train (context, not simulated: tubes without gas)
        if compact:
            train = [[(tx + 48, fy), (tx + 64, fy)], [(tx + 90, fy), (tx + 100, fy)], [(tx + 152, fy), (x_in, fy)]]
        else:
            train = [[(tx + 60, fy), (tx + 101, fy)], [(tx + 131, fy), (tx + 176, fy)], [(tx + 240, fy), (x_in, fy)]]
        for i, pts in enumerate(train):
            net.add(f"train_{i}", pts, lambda d: 0.0 * d["F0_feed"], lambda fr, F: None)
        for k in (0, 1):
            o = 1 - k
            fk = lambda d, v, k=k: d[f"F{k}_{v}"]  # noqa: E731
            net.add(f"noz_b_up{k}", [(cx[k], y_bot), (cx[k], node_b)], lambda d, k=k: -(d[f"F{k}_feed"] + d[f"F{k}_vent"] + d[f"F{k}_eq_bot"]),
                    lambda fr, F, k=k, o=o: bot(fr, k) if F > 0 else _mix([(pos(fr[f"F{k}_feed"]), yf), (pos(fr[f"F{k}_eq_bot"]), bot(fr, o))]))
            net.add(f"noz_b_lo{k}", [(cx[k], node_b), (cx[k], split_b)], lambda d, k=k: -(d[f"F{k}_feed"] + d[f"F{k}_vent"]),
                    lambda fr, F, k=k: bot(fr, k) if F > 0 else yf)
            net.add(f"feed_br{k}", [(cx[k], split_b), (cx[k] - dx, split_b), (cx[k] - dx, fy)], lambda d, k=k: -d[f"F{k}_feed"],
                    lambda fr, F, k=k: bot(fr, k) if F > 0 else yf)
            net.add(f"vent_br{k}", [(cx[k], split_b), (cx[k] + dx, split_b), (cx[k] + dx, vy)], lambda d, k=k: -d[f"F{k}_vent"],
                    lambda fr, F, k=k: bot(fr, k), halo=True)
            net.add(f"noz_t_lo{k}", [(cx[k], y_tops[k]), (cx[k], node_t)],
                    lambda d, k=k: -(d[f"F{k}_product"] + d[f"F{k}_purge"] + d[f"F{k}_eq_top"]),
                    lambda fr, F, k=k, o=o: top(fr, k) if F > 0 else _mix([(pos(fr[f"F{k}_product"]) + pos(fr[f"F{k}_purge"]), tank(fr)),
                                                                          (pos(fr[f"F{k}_eq_top"]), top(fr, o))]))
            net.add(f"noz_t_up{k}", [(cx[k], node_t), (cx[k], split_t)], lambda d, k=k: -(d[f"F{k}_product"] + d[f"F{k}_purge"]),
                    lambda fr, F, k=k: top(fr, k) if F > 0 else tank(fr))
            net.add(f"prod_br{k}", [(cx[k], split_t), (cx[k] - dx, split_t), (cx[k] - dx, py)], lambda d, k=k: -d[f"F{k}_product"],
                    lambda fr, F, k=k: top(fr, k) if F > 0 else tank(fr))
            net.add(f"purge_br{k}", [(cx[k], split_t), (cx[k] + dx, split_t), (cx[k] + dx, pq)], lambda d, k=k: -d[f"F{k}_purge"],
                    lambda fr, F: tank(fr), tube_w=11.0)
        net.add("purge_hdr_a", [(rx0 - 18, py), (rx0 - 18, pq), (cx[1] + dx, pq)], lambda d: d["F0_purge"] + d["F1_purge"],
                lambda fr, F: tank(fr), tube_w=11.0, halo=True)
        net.add("purge_hdr_b", [(cx[1] + dx, pq), (cx[0] + dx, pq)], lambda d: d["F0_purge"], lambda fr, F: tank(fr), tube_w=11.0, halo=True)
        net.finish()
        geo = dict(cx=cx, dx=dx, fy=fy, vy=vy, py=py, pq=pq, node_b=node_b, split_b=split_b, node_t=node_t, split_t=split_t,
                   mid=mid, tx=tx, compact=compact, x_in=x_in, vx=vx, rcx=rcx, receiver=(rx0, ry0, rx1, ry1))
        self._net_key, self._net_obj, self._geo = key, net, geo
        return net, geo

    def _pid(self, fig, fr):
        T, L = self.T, self.L
        ax = fig.add_axes([0, 0, 1, 1], zorder=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_axis_off()
        net, g = self._net()
        net.draw_tubes(ax)
        net.draw_gas(ax, fr, [T[sp] for sp in SPECIES])
        net.draw_chevrons(ax, fr)
        cx, dx, fy, vy, py, pq = g["cx"], g["dx"], g["fy"], g["vy"], g["py"], g["pq"]
        tx, x_in, vx = g["tx"], g["x_in"], g["vx"]
        mid = g["mid"]
        self._valve(ax, mid, g["node_b"], fr["open0_eq_bot"], vertical=False)
        self._valve(ax, mid, g["node_t"], fr["open0_eq_top"], vertical=False)
        # feed train (context, not simulated)
        if g["compact"]:
            self._compressor(ax, tx + 24, fy)
            self._vessel(ax, tx + 77, fy - 45, 26, 70, None)
            self._box_icon(ax, tx + 100, fy - 20, 52, 40, "dryer", small=True)
            self.text(fig, tx, vy + 18, "compressor, air receiver, dryer · 7.5 barg, dew point +3 °C (context only)",
                      size=10, color=T["muted"])
        else:
            self._compressor(ax, tx + 34, fy)
            self._vessel(ax, tx + 116, fy - 50, 30, 78, None)
            self._box_icon(ax, tx + 176, fy - 22, 64, 44, "dryer")
            self.text(fig, tx, fy + 34, "compressor, air receiver, dryer\n7.5 barg, dew point +3 °C (context only)",
                      size=10, color=T["muted"], linespacing=1.3)
        self._box_icon(ax, vx, vy - 15, 56, 30, "silencer", small=True)
        self.text(fig, vx + 64, vy - 9, "vent", size=11, color=T["muted"])
        rx0, ry0, rx1, ry1 = g["receiver"]
        rcx = g["rcx"]
        self._vessel(ax, rcx, ry0, rx1 - rx0, ry1 - ry0, "N$_2$ receiver")
        self._regulator(ax, rcx, ry1 + 30, vertical=True)
        self.text(fig, rcx + 16, ry1 + 54, "N$_2$ product\n30 SLPM\n6 barg", size=11, linespacing=1.25)
        # valves on the branches
        split_b, split_t = g["split_b"], g["split_t"]
        vb = split_b + 0.5 * (fy - split_b)
        vv = split_b + 0.62 * (vy - split_b)
        vt = py + 0.55 * (split_t - py)
        vtp = pq + 0.5 * (split_t - pq)
        for k in (0, 1):
            self._valve(ax, cx[k] - dx, vb, fr[f"open{k}_feed"])
            self._valve(ax, cx[k] + dx, vv, fr[f"open{k}_vent"])
            self._valve(ax, cx[k] - dx, vt, fr[f"open{k}_product"])
            self._valve(ax, cx[k] + dx, vtp, fr[f"open{k}_purge"], size=10, needle=True)

    def _compressor(self, ax, x, y):
        T = self.T
        ax.add_patch(plt.Circle((x, y), 24, fc=T["bg"], ec=T["muted"], lw=1.6 * self.s, zorder=5))
        ax.add_patch(Polygon([(x - 14, y - 16), (x - 14, y + 16), (x + 18, y + 7), (x + 18, y - 7)], closed=True,
                             fc="none", ec=T["muted"], lw=1.4 * self.s, zorder=6))

    def _vessel(self, ax, xc, y0, w, h, label):
        T = self.T
        ax.add_patch(FancyBboxPatch((xc - w / 2, y0), w, h, boxstyle=f"round,pad=0,rounding_size={w / 2:.1f}",
                                    fc=T["bg"], ec=T["muted"], lw=1.6 * self.s, zorder=5))
        if label:
            ax.text(xc, y0 - 8, label, ha="center", va="bottom", fontsize=self.fs(10.5), color=T["muted"])

    def _box_icon(self, ax, x, y, w, h, label, small=False):
        T = self.T
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6", fc=T["bg"], ec=T["muted"],
                                    lw=1.6 * self.s, zorder=5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=self.fs(9 if small else 10.5),
                color=T["muted"], zorder=6)

    def _regulator(self, ax, x, y, vertical=False):
        T = self.T
        self._valve(ax, x, y, True, vertical=vertical, size=9)
        if vertical:
            ax.plot([x, x + 18], [y, y], color=T["muted"], lw=1.2 * self.s, zorder=6)
            ax.add_patch(plt.Circle((x + 23, y), 5, fc="none", ec=T["muted"], lw=1.2 * self.s, zorder=6))
        else:
            ax.plot([x, x], [y, y - 18], color=T["muted"], lw=1.2 * self.s, zorder=6)
            ax.add_patch(plt.Circle((x, y - 23), 5, fc="none", ec=T["muted"], lw=1.2 * self.s, zorder=6))

    # ---------------------------------------------------------------- panels
    def _axis_style(self, a, title=None):
        a.tick_params(length=0, labelsize=self.fs(10.5))
        if title:
            a.set_title(title, loc="left", fontsize=self.fs(12.5), pad=7 * self.s)

    def _o2_panel(self, a, curves, show_y):
        T = self.T
        for y, ls in curves:
            a.plot(100 * y[IO2], self.z, color=T["O2"], lw=2.0 * self.s, ls=ls)
        a.axhspan(0, self.z_al, color=T["highlight"], lw=0, zorder=0)
        a.set(xlim=(0, self.o2_xmax), ylim=(0, 1), xticks=list(range(0, self.o2_xmax + 1, 10 if self.o2_xmax <= 40 else 20)))
        a.set_yticks([0, 0.5, 1] if show_y else [])
        if show_y:
            a.set_ylabel("height  z / L", fontsize=self.fs(11.5))
        a.set_xlabel("O$_2$ (%)", fontsize=self.fs(11))

    def _trace_panel(self, a, curves):
        T = self.T
        for y, ls in curves:
            for i in (IAR, ICO2, IH2O):
                a.plot(np.maximum(1e6 * y[i], 1e-3), self.z, color=T[SPECIES[i]], lw=2.0 * self.s, ls=ls)
        a.axhspan(0, self.z_al, color=T["highlight"], lw=0, zorder=0)
        a.set(xscale="log", xlim=(0.1, 3e4), ylim=(0, 1), yticks=[])
        a.set_xticks([1, 100, 1e4])
        a.set_xticklabels(["1", "100", "10$^4$"])
        a.set_xlabel("ppm (log)", fontsize=self.fs(11))

    def _trace_key(self, fig, box):
        T = self.T
        x, y = box[0], box[1] - 30
        for j, (i, lab) in enumerate(((IAR, "Ar"), (ICO2, "CO$_2$"), (IH2O, "H$_2$O"))):
            xx = x + j * 56
            fig.lines.append(plt.Line2D([self.fx(xx + 5)], [self.fy(y + 8)], marker="o", ms=6 * self.s,
                                        color=T[SPECIES[i]], mec=T["bg"], transform=fig.transFigure, figure=fig))
            self.text(fig, xx + 13, y, lab, size=10.5, color=T["ink2"])

    def _profiles(self, fig, fr):
        if self.L["prof"] == "shared":
            b1, b2 = self.L["prof_boxes"]
            curves = [(fr["y0"], "-"), (fr["y1"], "--")]
            a = self.box_axes(fig, b1)
            self._o2_panel(a, curves, show_y=True)
            self._axis_style(a, "O$_2$ in the gas")
            b = self.box_axes(fig, b2)
            self._trace_panel(b, curves)
            self._axis_style(b)
            self._trace_key(fig, b2)
            self.text(fig, b1[0], b1[3] + 46, "solid: bed 1   dashed: bed 2", size=10.5, color=self.T["muted"])
            return
        for k in (0, 1):
            b1, b2 = self.L["prof_boxes"][k]
            curves = [(fr[f"y{k}"], "-")]
            a = self.box_axes(fig, b1)
            self._o2_panel(a, curves, show_y=(k == 0))
            self._axis_style(a, f"bed {k + 1} gas")
            b = self.box_axes(fig, b2)
            self._trace_panel(b, curves)
            self._axis_style(b)
            if k == 0:
                self._trace_key(fig, b2)

    def _timing(self, fig, fr):
        T = self.T
        x0, y0, x1, y1 = self.L["timing"]
        a = self.box_axes(fig, (x0 + 92, y0 + 44, x1 - 22, y1))
        self.text(fig, x0, y0, "Steps and valves", size=13, fontweight="bold")
        self.text(fig, x0, y0 + 21, "bars: valve open (upper bed 1, lower bed 2); hatched: product back-flow",
                  size=10, color=T["muted"])
        rows = [("bed 1", None), ("bed 2", None), ("feed", "feed"), ("product", "product"), ("purge", "purge"),
                ("vent", "vent"), ("equalize", "eq_top")]
        nrow = len(rows)
        d = self.d
        frac = d["t_cycle"] / self.t_cycle
        for r, (lab, key) in enumerate(rows):
            y = nrow - 1 - r
            a.text(-0.02, y, lab, ha="right", va="center", fontsize=self.fs(11), color=T["ink2"],
                   transform=a.get_yaxis_transform())
            if key is None:
                k = r
                for s_ in range(4):
                    xa, xb = self.starts[s_], self.starts[s_ + 1]
                    if s_ % 2 == 0:
                        name, shade = None, T["highlight"]
                    else:
                        prod = (s_ == 1) == (k == 0)
                        name = "production" if prod else "blowdown + purge"
                        shade = T["axis"] if prod else T["grid"]
                    a.add_patch(Rectangle((xa, y - 0.38), xb - xa, 0.76, fc=shade, ec=T["bg"], lw=1.0 * self.s))
                    if name:
                        a.text(0.5 * (xa + xb), y, name, ha="center", va="center", fontsize=self.fs(10), color=T["ink"])
                for w0, w1 in self._backflow[k]:
                    a.add_patch(Rectangle((w0, y - 0.38), max(w1 - w0, 0.004), 0.76, fc="none", ec=T["N2"],
                                          hatch="////", lw=0.8 * self.s))
                continue
            for k in (0, 1):
                m = np.asarray(d[f"open{k}_{key}"], bool)
                yy = y + (0.17 if k == 0 else -0.17)
                if m.any():
                    idx = np.nonzero(m)[0]
                    for s in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                        xa, xb = frac[s[0]], frac[min(s[-1] + 1, len(frac) - 1)]
                        if xb < xa:
                            xb = 1.0
                        a.add_patch(Rectangle((xa, yy - 0.12), xb - xa, 0.24, fc=T["ink2"] if k == 0 else T["muted"], lw=0))
        a.axvline(fr["frac"], color=T["ink"], lw=1.8 * self.s)
        a.set(xlim=(0, 1), ylim=(-0.6, nrow - 0.4), xticks=[0, 0.25, 0.5, 0.75, 1.0], yticks=[])
        a.set_xlabel("t / t$_{cycle}$", fontsize=self.fs(11), labelpad=2)
        a.grid(False)
        a.spines["left"].set_visible(False)
        self._axis_style(a)

    def _pressure(self, fig, fr):
        T = self.T
        x0, y0, x1, y1 = self.L["pressure"]
        a = self.box_axes(fig, (x0 + 40, y0 + 30, x1 - 80, y1))
        self.text(fig, x0, y0, f"Pressure ({self.unit})", size=13, fontweight="bold")
        d = self.d
        tt = d["trace_t"] / self.t_cycle
        lines = (("bed 1", d["trace_P0"], T["ink"], "-", "P0"), ("bed 2", d["trace_P1"], T["ink2"], "--", "P1"),
                 ("receiver", d["trace_PT"], T["muted"], "-", "p_tank"))
        for lab, p, col, ls, key in lines:
            a.plot(tt, self.p_disp(p), color=col, lw=1.8 * self.s, ls=ls)
            a.plot([fr["frac"]], [self.p_disp(fr[key])], "o", ms=6 * self.s, color=col, mec=T["bg"], mew=1.2 * self.s)
        ends = spread([self.p_disp(p[-1]) for _, p, *_ in lines], 0.9)
        for (lab, *_), yl in zip(lines, ends):
            a.text(1.02, yl, lab, fontsize=self.fs(10), color=T["ink2"], va="center", transform=a.get_yaxis_transform())
        a.set(xlim=(0, 1), xticks=[0, 0.5, 1])
        a.grid(axis="x", visible=False)
        self._axis_style(a)

    def _o2(self, fig, fr):
        T = self.T
        x0, y0, x1, y1 = self.L["o2"]
        a = self.box_axes(fig, (x0 + 52, y0 + 30, x1 - 80, y1))
        self.text(fig, x0, y0, "Product O$_2$ (ppm)", size=13, fontweight="bold")
        self.text(fig, x0 + 190, y0 + 2, "receiver outlet", size=10, color=T["muted"])
        d = self.d
        tt = d["trace_t"] / self.t_cycle
        o2 = 1e6 * d["trace_yT"][:, IO2]
        a.plot(tt, o2, color=T["O2"], lw=2.0 * self.s)
        a.axhline(5000, color=T["muted"], lw=1.0 * self.s, ls=":")
        a.text(1.02, 5000, "spec", fontsize=self.fs(10), color=T["muted"], va="center", transform=a.get_yaxis_transform())
        a.plot([fr["frac"]], [1e6 * fr["y_tank"][IO2]], "o", ms=6 * self.s, color=T["O2"], mec=T["bg"], mew=1.2 * self.s)
        lo, hi = min(o2.min(), 5000), max(o2.max(), 5000)
        a.set(xlim=(0, 1), ylim=(lo - 0.15 * (hi - lo) - 50, hi + 0.15 * (hi - lo) + 50), xticks=[0, 0.5, 1])
        a.grid(axis="x", visible=False)
        self._axis_style(a)

    def _readouts(self, fig, fr):
        T, m = self.T, self.met
        x, y = self.L["readouts"]
        yT = fr["y_tank"]
        o2 = 1e6 * yT[IO2]
        rows = [
            ("O$_2$", f"{o2:,.0f} ppm   (spec ≤ 5,000)"),
            ("purity", f"{100 * (1 - yT[IO2]):.2f} % N$_2$ + Ar"),
            ("N$_2$ only", f"{100 * yT[IN2]:.2f} %   (Ar {100 * yT[IAR]:.2f} %)"),
            ("CO$_2$, H$_2$O", f"{1e6 * yT[ICO2]:.2g} ppm, {1e6 * yT[IH2O]:.2g} ppm"),
            ("recovery", f"N$_2$ {100 * m['recovery_N2']:.1f} %,   air factor {m['air_factor']:.2f}"),
            ("productivity", f"{m['productivity_Nm3_h_m3']:.0f} Nm$^3$/h per m$^3$ CMS"),
        ]
        self.text(fig, x, y, "Product 30 SLPM (21.1 °C, 1 atm)", size=13, fontweight="bold")
        for i, (k, v) in enumerate(rows):
            self.text(fig, x, y + 28 + i * 21, k, size=11.5, color=T["ink2"])
            self.text(fig, x + 118, y + 28 + i * 21, v, size=11.5)

    def _legend(self, fig):
        T = self.T
        x, y = self.L["legend"]
        for i, s in enumerate(SPECIES):
            yy = y + i * 20
            fig.lines.append(plt.Line2D([self.fx(x + 7)], [self.fy(yy + 8)], marker="o", ms=7.5 * self.s, color=T[s],
                                        mec=T["bg"], transform=fig.transFigure, figure=fig))
            self.text(fig, x + 20, yy, f"{SPECIES_TEX[s]}  {SPECIES_ROLE[s]}", size=10.5)
        yr = y + 5 * 20 + 6
        for j, (rmp, lab) in enumerate(((self.ramp_cms, "CMS: O$_2$ held, q / q$_{feed}$"),
                                        (self.ramp_al, "alumina: H$_2$O held, q / q$_{feed}$"))):
            xx, yy = x, yr + j * 40
            self.text(fig, xx, yy, lab, size=10, color=T["ink2"])
            ax = fig.add_axes([self.fx(xx + 200), self.fy(yy + 12), 90 / W, 9 / H])
            ax.imshow(rmp[None], aspect="auto", extent=(0, 1, 0, 1))
            ax.set_xticks([0, 1])
            ax.set_yticks([])
            ax.tick_params(length=0, labelsize=self.fs(9), pad=1)
            for sp_ in ax.spines.values():
                sp_.set_visible(False)

    def _title(self, fig, fr):
        T = self.T
        x, y = self.L["title"]
        tx = self.cfg.get("text", {})
        self.text(fig, x, y, tx.get("title", "PSA nitrogen generator"), size=26, fontweight="bold")
        self.text(fig, x, y + 40, tx.get("subtitle", ""), size=14.5, color=T["ink2"])
        self.text(fig, x, y + 64, f"t / t$_{{cycle}}$ = {fr['frac']:.3f}   (t$_{{cycle}}$ = {self.t_cycle:.0f} s)", size=13.5,
                  color=T["ink2"], family="DejaVu Sans Mono")
