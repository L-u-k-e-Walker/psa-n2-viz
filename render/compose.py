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
import matplotlib.ticker  # noqa: E402
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
    "D": dict(  # conventional twin-tower P&ID: manifolds between the beds, a profile chart beside each bed
        name="twin towers with manifolds", pid="manifold", title=(36, 24), legend=(1330, 866),
        beds=[(200, 320, 310, 730), (1010, 320, 1120, 730)], prof="per_bed_all", label_side=["left", "right"],
        prof_boxes=[(380, 320, 600, 730), (720, 320, 940, 730)],
        # manifold levels (design px, y from the top), from the beds outwards
        eq_top_y=272, purge_y=222, prod_y=172, recv_line_y=124, eq_bot_y=792, vent_y=852, feed_y=912,
        feed_out_y=992, vent_out_y=1024, valve_dx=175, tee_feed_x=600, tee_vent_x=720, tee_prod_x=600, tee_purge_x=720,
        train_x=20, train="compact", receiver=(1150, 96, 1204, 214), silencer_x=1130,
        timing=(1330, 52, 1890, 300), pressure=(1330, 352, 1890, 488), o2=(1330, 540, 1890, 656),
        readouts=(1330, 696), footnote=(36, 1072)),
    "E": dict(  # twin towers, two 3-way valves per end (user, v06 review); a profile chart beside each bed
        name="twin towers, 3-way valves", pid="threeway", title=(36, 24), legend=(1330, 866),
        beds=[(200, 344, 310, 744), (1010, 344, 1120, 744)], prof="per_bed_all", label_side=["left", "right"],
        prof_boxes=[(380, 344, 600, 744), (720, 344, 940, 744)],
        v3_top_y=294, purge_y=236, recv_line_y=150, v3_bot_y=832, vent_y=900, feed_out_y=978, vent_out_y=1024,
        tee_feed_x=600, tee_vent_x=720, tee_prod_x=600, tee_purge_x=720,
        train_x=20, train="compact", receiver=(1196, 92, 1246, 200), silencer_x=1130,
        gauges=[(150, 212), (1144, 212)], gauge_r=30,
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
                fig.text(0.985, 0.045, watermark, ha="right", va="bottom", fontsize=self.fs(26), color=T["muted"],
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
    def _valve(self, ax, x, y, open_, vertical=True, size=13, needle=False, actuator=True):
        style = self.cfg.get("layout", {}).get("valve_style", "plain")
        if style == "ball":
            return self._valve_ball(ax, x, y, open_, vertical, size, needle)
        T = self.T
        s = size
        if style == "isa" and not needle and actuator:
            self._actuator(ax, x, y, "l" if vertical else "u", size)
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

    def _net_manifold(self):
        """Conventional twin-tower piping: horizontal manifolds between the bed verticals, one valve per
        bed on each, the outlet/inlet tee in the middle (the equalization lines have one valve)."""
        key = ("manifold", None if self.nozzles is None else tuple(map(tuple, self.nozzles)))
        if getattr(self, "_net_key", "unset") == key:
            return self._net_obj, self._geo
        L, T = self.L, self.T
        beds = [self.bed_geometry(k) for k in (0, 1)]
        cx = [0.5 * (b[0] + b[2]) for b in beds]
        mid = 0.5 * (cx[0] + cx[1])
        if self.nozzles is not None:
            y_tops = [nz[0] for nz in self.nozzles]
            y_bots = [nz[1] for nz in self.nozzles]
        else:
            y_tops = [b[1] - 14 for b in beds]
            y_bots = [b[3] + 14 for b in beds]
        et, pu, pr, rl = L["eq_top_y"], L["purge_y"], L["prod_y"], L["recv_line_y"]
        eb, ve, fe = L["eq_bot_y"], L["vent_y"], L["feed_y"]
        fo, vo = L["feed_out_y"], L["vent_out_y"]
        txf, txv, txp, txq = L["tee_feed_x"], L["tee_vent_x"], L["tee_prod_x"], L["tee_purge_x"]
        rx0, ry0, rx1, ry1 = L["receiver"]
        rcx = 0.5 * (rx0 + rx1)
        sx = L["silencer_x"]
        tx = L["train_x"]
        x_in = tx + 170
        yf = self.cyc.pl.yf
        mcfg = self.cfg.get("molecules", {}).get("weights", {})
        w = [mcfg.get(sp, 1.0) for sp in SPECIES]
        pcfg = self.cfg.get("pipes", {})
        net = PipeNet(self.d, yf, w, T, self.s, tube_w=pcfg.get("tube_px", 16.0), spacing=pcfg.get("dot_spacing_px", 16.0),
                      vmax=pcfg.get("vmax_px_s", 260.0), f_ref=pcfg.get("f_ref_mol_s", 0.05))
        bot = lambda fr, k: fr[f"y{k}"][:, 0]  # noqa: E731
        top = lambda fr, k: fr[f"y{k}"][:, -1]  # noqa: E731
        tank = lambda fr: fr["y_tank"]  # noqa: E731

        def pos(x):
            return np.maximum(x, 0.0)

        # bed verticals, split at each manifold level (segment flows = everything passing through it)
        for k in (0, 1):
            o = 1 - k
            F = lambda d, names, k=k: sum(d[f"F{k}_{n}"] for n in names)  # noqa: E731
            # bottom: nozzle -> eq_bot -> vent -> feed (down = away from the bed)
            net.add(f"vb0_{k}", [(cx[k], y_bots[k]), (cx[k], eb)], lambda d, F=F: -F(d, ("feed", "vent", "eq_bot")),
                    lambda fr, Fv, k=k, o=o: bot(fr, k) if Fv > 0 else _mix([(pos(fr[f"F{k}_feed"]), yf), (pos(fr[f"F{k}_eq_bot"]), bot(fr, o))]))
            net.add(f"vb1_{k}", [(cx[k], eb), (cx[k], ve)], lambda d, F=F: -F(d, ("feed", "vent")),
                    lambda fr, Fv, k=k: bot(fr, k) if Fv > 0 else yf)
            net.add(f"vb2_{k}", [(cx[k], ve), (cx[k], fe)], lambda d, F=F: -F(d, ("feed",)),
                    lambda fr, Fv, k=k: bot(fr, k) if Fv > 0 else yf)
            # top: nozzle -> eq_top -> purge -> product (up = away from the bed)
            net.add(f"vt0_{k}", [(cx[k], y_tops[k]), (cx[k], et)], lambda d, F=F: -F(d, ("product", "purge", "eq_top")),
                    lambda fr, Fv, k=k, o=o: top(fr, k) if Fv > 0 else _mix([(pos(fr[f"F{k}_product"]) + pos(fr[f"F{k}_purge"]), tank(fr)),
                                                                            (pos(fr[f"F{k}_eq_top"]), top(fr, o))]))
            net.add(f"vt1_{k}", [(cx[k], et), (cx[k], pu)], lambda d, F=F: -F(d, ("product", "purge")),
                    lambda fr, Fv, k=k: top(fr, k) if Fv > 0 else tank(fr))
            net.add(f"vt2_{k}", [(cx[k], pu), (cx[k], pr)], lambda d, F=F: -F(d, ("product",)),
                    lambda fr, Fv, k=k: top(fr, k) if Fv > 0 else tank(fr))
        # manifolds: left half carries bed 1's flow, right half bed 2's (+ = towards the tee / bed 2)
        net.add("eq_bot", [(cx[0], eb), (cx[1], eb)], lambda d: d["F1_eq_bot"], lambda fr, F: bot(fr, 0) if F > 0 else bot(fr, 1))
        net.add("eq_top", [(cx[0], et), (cx[1], et)], lambda d: d["F1_eq_top"], lambda fr, F: top(fr, 0) if F > 0 else top(fr, 1))
        net.add("vent_l", [(cx[0], ve), (txv, ve)], lambda d: -d["F0_vent"], lambda fr, F: bot(fr, 0))
        net.add("vent_r", [(cx[1], ve), (txv, ve)], lambda d: -d["F1_vent"], lambda fr, F: bot(fr, 1))
        net.add("feed_l", [(txf, fe), (cx[0], fe)], lambda d: d["F0_feed"], lambda fr, F: yf if F > 0 else bot(fr, 0))
        net.add("feed_r", [(txf, fe), (cx[1], fe)], lambda d: d["F1_feed"], lambda fr, F: yf if F > 0 else bot(fr, 1))
        net.add("prod_l", [(cx[0], pr), (txp, pr)], lambda d: -d["F0_product"], lambda fr, F: top(fr, 0) if F > 0 else tank(fr))
        net.add("prod_r", [(cx[1], pr), (txp, pr)], lambda d: -d["F1_product"], lambda fr, F: top(fr, 1) if F > 0 else tank(fr))
        net.add("purge_l", [(txq, pu), (cx[0], pu)], lambda d: d["F0_purge"], lambda fr, F: tank(fr), tube_w=11.0)
        net.add("purge_r", [(txq, pu), (cx[1], pu)], lambda d: d["F1_purge"], lambda fr, F: tank(fr), tube_w=11.0)
        # risers / outlets
        prod_mix = lambda fr, F: (_mix([(pos(-fr["F0_product"]), top(fr, 0)), (pos(-fr["F1_product"]), top(fr, 1))])  # noqa: E731
                                  if F > 0 else tank(fr))
        net.add("prod_up", [(txp, pr), (txp, rl), (txq, rl)], lambda d: -(d["F0_product"] + d["F1_product"]), prod_mix)
        net.add("recv_in", [(txq, rl), (rx0, rl)],
                lambda d: -(d["F0_product"] + d["F1_product"]) - (d["F0_purge"] + d["F1_purge"]),
                lambda fr, F: prod_mix(fr, 1.0) if F > 0 else tank(fr))
        net.add("purge_down", [(txq, rl), (txq, pu)], lambda d: d["F0_purge"] + d["F1_purge"], lambda fr, F: tank(fr),
                tube_w=11.0, halo=True)
        net.add("product_out", [(rcx, ry1), (rcx, ry1 + 72)], lambda d: d["F_product_out"],
                lambda fr, F: tank(fr))
        net.add("feed_in", [(x_in, fo), (txf, fo), (txf, fe)], lambda d: d["F0_feed"] + d["F1_feed"], lambda fr, F: yf)
        net.add("vent_out", [(txv, ve), (txv, vo), (sx, vo)], lambda d: -(d["F0_vent"] + d["F1_vent"]),
                lambda fr, F: _mix([(pos(-fr["F0_vent"]), bot(fr, 0)), (pos(-fr["F1_vent"]), bot(fr, 1))]) if F > 0 else bot(fr, 0),
                halo=True)
        for i, pts in enumerate([[(tx + 48, fo), (tx + 64, fo)], [(tx + 90, fo), (tx + 100, fo)], [(tx + 152, fo), (x_in, fo)]]):
            net.add(f"train_{i}", pts, lambda d: 0.0 * d["F0_feed"], lambda fr, F: None)
        net.finish()
        vdx = L["valve_dx"]
        geo = dict(cx=cx, mid=mid, et=et, pu=pu, pr=pr, eb=eb, ve=ve, fe=fe, fo=fo, vo=vo, tx=tx, sx=sx, rcx=rcx,
                   receiver=(rx0, ry0, rx1, ry1), valves_x=[cx[0] + vdx, cx[1] - vdx])
        self._net_key, self._net_obj, self._geo = key, net, geo
        return net, geo

    def _pid_manifold(self, fig, fr):
        T = self.T
        ax = fig.add_axes([0, 0, 1, 1], zorder=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_axis_off()
        net, g = self._net_manifold()
        net.draw_tubes(ax)
        net.draw_gas(ax, fr, [T[sp] for sp in SPECIES])
        net.draw_chevrons(ax, fr)
        vx = g["valves_x"]
        for k in (0, 1):
            self._valve(ax, vx[k], g["fe"], fr[f"open{k}_feed"], vertical=False)
            self._valve(ax, vx[k], g["ve"], fr[f"open{k}_vent"], vertical=False)
            self._valve(ax, vx[k], g["pr"], fr[f"open{k}_product"], vertical=False)
            self._valve(ax, vx[k], g["pu"], fr[f"open{k}_purge"], vertical=False, size=11, needle=True)
        self._valve(ax, g["mid"], g["eb"], fr["open0_eq_bot"], vertical=False)
        self._valve(ax, g["mid"], g["et"], fr["open0_eq_top"], vertical=False)
        # line names at the left end of each manifold (between bed 1 and its valve)
        for y, lab in ((g["et"], "equalization"), (g["pu"], "purge"), (g["pr"], "product"),
                       (g["eb"], "equalization"), (g["ve"], "exhaust"), (g["fe"], "feed")):
            self.text(fig, g["cx"][0] + 14, y - 12, lab, size=10, color=T["muted"], va="bottom")
        tx, fo, vo = g["tx"], g["fo"], g["vo"]
        self._compressor(ax, tx + 24, fo)
        self._vessel(ax, tx + 77, fo - 45, 26, 70, None)
        self._box_icon(ax, tx + 100, fo - 20, 52, 40, "dryer", small=True)
        self.text(fig, tx, fo + 34, "compressor, air receiver, dryer · 7.5 barg, dew point +3 °C (context only)",
                  size=10, color=T["muted"])
        self._box_icon(ax, g["sx"], vo - 15, 56, 30, "silencer", small=True)
        self.text(fig, g["sx"] + 64, vo - 9, "vent", size=11, color=T["muted"])
        rx0, ry0, rx1, ry1 = g["receiver"]
        self._vessel(ax, g["rcx"], ry0, rx1 - rx0, ry1 - ry0, "N$_2$ receiver")
        self._regulator(ax, g["rcx"], ry1 + 30, vertical=True)
        self.text(fig, g["rcx"] + 34, ry1 + 16, "N$_2$ product\n30 SLPM, 6 barg", size=10.5, linespacing=1.25)

    def _net_threeway(self):
        """Two 3-way valves per end. Bottom: bed <-> feed line | exhaust; one feed-inlet valve on the feed tee.
        Top: bed <-> product line | purge line; an outlet valve on the riser to the receiver, a needle valve on
        the purge supply. The model's equalization valves are drawn as the path through both 3-way valves
        with the inlet (bottom) / outlet (top) valve shut. Segment flows are the model's valve flows."""
        key = ("threeway", None if self.nozzles is None else tuple(map(tuple, self.nozzles)))
        if getattr(self, "_net_key", "unset") == key:
            return self._net_obj, self._geo
        L, T = self.L, self.T
        beds = [self.bed_geometry(k) for k in (0, 1)]
        cx = [0.5 * (b[0] + b[2]) for b in beds]
        if self.nozzles is not None:
            y_tops = [nz[0] for nz in self.nozzles]
            y_bots = [nz[1] for nz in self.nozzles]
        else:
            y_tops = [b[1] - 14 for b in beds]
            y_bots = [b[3] + 14 for b in beds]
        vt, pu, rl = L["v3_top_y"], L["purge_y"], L["recv_line_y"]
        vb, ve, fo, vo = L["v3_bot_y"], L["vent_y"], L["feed_out_y"], L["vent_out_y"]
        txf, txv, txp, txq = L["tee_feed_x"], L["tee_vent_x"], L["tee_prod_x"], L["tee_purge_x"]
        rx0, ry0, rx1, ry1 = L["receiver"]
        rcx = 0.5 * (rx0 + rx1)
        sx, tx = L["silencer_x"], L["train_x"]
        x_in = tx + 170
        yf = self.cyc.pl.yf
        mcfg = self.cfg.get("molecules", {}).get("weights", {})
        w = [mcfg.get(sp, 1.0) for sp in SPECIES]
        pcfg = self.cfg.get("pipes", {})
        net = PipeNet(self.d, yf, w, T, self.s, tube_w=pcfg.get("tube_px", 16.0), spacing=pcfg.get("dot_spacing_px", 16.0),
                      vmax=pcfg.get("vmax_px_s", 260.0), f_ref=pcfg.get("f_ref_mol_s", 0.05))
        bot = lambda fr, k: fr[f"y{k}"][:, 0]  # noqa: E731
        top = lambda fr, k: fr[f"y{k}"][:, -1]  # noqa: E731
        tank = lambda fr: fr["y_tank"]  # noqa: E731

        def pos(x):
            return np.maximum(x, 0.0)

        # lines first, risers that cross them later (with a halo)
        net.add("exh_l", [(cx[0], vb), (cx[0], ve), (txv, ve)], lambda d: -d["F0_vent"], lambda fr, F: bot(fr, 0))
        net.add("exh_r", [(cx[1], vb), (cx[1], ve), (txv, ve)], lambda d: -d["F1_vent"], lambda fr, F: bot(fr, 1))
        net.add("purge_l", [(txq, pu), (cx[0], pu), (cx[0], vt)], lambda d: d["F0_purge"], lambda fr, F: tank(fr), tube_w=11.0)
        net.add("purge_r", [(txq, pu), (cx[1], pu), (cx[1], vt)], lambda d: d["F1_purge"], lambda fr, F: tank(fr), tube_w=11.0)
        for k in (0, 1):
            o = 1 - k
            net.add(f"bed_bot{k}", [(cx[k], y_bots[k]), (cx[k], vb)],
                    lambda d, k=k: -(d[f"F{k}_feed"] + d[f"F{k}_vent"] + d[f"F{k}_eq_bot"]),
                    lambda fr, F, k=k, o=o: bot(fr, k) if F > 0 else _mix([(pos(fr[f"F{k}_feed"]), yf), (pos(fr[f"F{k}_eq_bot"]), bot(fr, o))], bot(fr, k)))
            net.add(f"bed_top{k}", [(cx[k], y_tops[k]), (cx[k], vt)],
                    lambda d, k=k: -(d[f"F{k}_product"] + d[f"F{k}_purge"] + d[f"F{k}_eq_top"]),
                    lambda fr, F, k=k, o=o: top(fr, k) if F > 0 else _mix([(pos(fr[f"F{k}_product"]) + pos(fr[f"F{k}_purge"]), tank(fr)),
                                                                          (pos(fr[f"F{k}_eq_top"]), top(fr, o))], top(fr, k)))
        # feed / bottom-equalization line: halves carry bed k's feed + equalization flow (+ = towards bed 2)
        net.add("feed_l", [(cx[0], vb), (txf, vb)], lambda d: -(d["F0_feed"] + d["F0_eq_bot"]),
                lambda fr, F: bot(fr, 0) if F > 0 else _mix([(pos(fr["F0_feed"]), yf), (pos(fr["F0_eq_bot"]), bot(fr, 1))], yf))
        net.add("feed_r", [(txf, vb), (cx[1], vb)], lambda d: d["F1_feed"] + d["F1_eq_bot"],
                lambda fr, F: _mix([(pos(fr["F1_feed"]), yf), (pos(fr["F1_eq_bot"]), bot(fr, 0))], yf) if F > 0 else bot(fr, 1))
        # product / top-equalization line
        net.add("prod_l", [(cx[0], vt), (txp, vt)], lambda d: -(d["F0_product"] + d["F0_eq_top"]),
                lambda fr, F: top(fr, 0) if F > 0 else _mix([(pos(fr["F0_product"]), tank(fr)), (pos(fr["F0_eq_top"]), top(fr, 1))], tank(fr)))
        net.add("prod_r", [(txp, vt), (cx[1], vt)], lambda d: d["F1_product"] + d["F1_eq_top"],
                lambda fr, F: _mix([(pos(fr["F1_product"]), tank(fr)), (pos(fr["F1_eq_top"]), top(fr, 0))], tank(fr)) if F > 0 else top(fr, 1))
        prod_mix = lambda fr, F: (_mix([(pos(-fr["F0_product"]), top(fr, 0)), (pos(-fr["F1_product"]), top(fr, 1))], tank(fr))  # noqa: E731
                                  if F > 0 else tank(fr))
        net.add("prod_up", [(txp, vt), (txp, rl), (txq, rl)], lambda d: -(d["F0_product"] + d["F1_product"]), prod_mix, halo=True)
        net.add("recv_in", [(txq, rl), (rx0, rl)], lambda d: -(d["F0_product"] + d["F1_product"]) - (d["F0_purge"] + d["F1_purge"]),
                lambda fr, F: prod_mix(fr, 1.0) if F > 0 else tank(fr))
        net.add("purge_down", [(txq, rl), (txq, pu)], lambda d: d["F0_purge"] + d["F1_purge"], lambda fr, F: tank(fr), tube_w=11.0)
        net.add("product_out", [(rcx, ry1), (rcx, ry1 + 70)], lambda d: d["F_product_out"], lambda fr, F: tank(fr))
        net.add("feed_in", [(x_in, fo), (txf, fo), (txf, vb)], lambda d: d["F0_feed"] + d["F1_feed"], lambda fr, F: yf, halo=True)
        net.add("vent_out", [(txv, ve), (txv, vo), (sx, vo)], lambda d: -(d["F0_vent"] + d["F1_vent"]),
                lambda fr, F: _mix([(pos(-fr["F0_vent"]), bot(fr, 0)), (pos(-fr["F1_vent"]), bot(fr, 1))], bot(fr, 0)) if F > 0 else bot(fr, 0))
        for i, pts in enumerate([[(tx + 48, fo), (tx + 64, fo)], [(tx + 90, fo), (tx + 100, fo)], [(tx + 152, fo), (x_in, fo)]]):
            net.add(f"train_{i}", pts, lambda d: 0.0 * d["F0_feed"], lambda fr, F: None)
        # gauge stems: from the bed's top line to the dial (no gas drawn)
        for k, (gx, gy) in enumerate(L["gauges"]):
            ys = y_tops[k] - 12
            side = 1 if k == 0 else -1
            xr = gx + side * (L["gauge_r"] + 20)
            net.add(f"gauge_stem{k}", [(cx[k], ys), (xr, ys), (xr, gy), (gx + side * L["gauge_r"], gy)],
                    lambda d: 0.0 * d["F0_feed"], lambda fr, F: None, tube_w=7.0)
        net.finish()
        geo = dict(cx=cx, vt=vt, pu=pu, rl=rl, vb=vb, ve=ve, fo=fo, vo=vo, tx=tx, sx=sx, rcx=rcx, txf=txf, txp=txp, txq=txq,
                   receiver=(rx0, ry0, rx1, ry1), y_tops=y_tops)
        self._net_key, self._net_obj, self._geo = key, net, geo
        return net, geo

    def _actuator(self, ax, x, y, side, size):
        """Pneumatic diaphragm actuator (ISA): a stem from the body and a dome."""
        T = self.T
        vec = {"u": (0, -1), "d": (0, 1), "l": (-1, 0), "r": (1, 0)}
        ux, uy = vec[side]
        L = 1.9 * size
        ax.plot([x, x + ux * L], [y, y + uy * L], color=T["ink2"], lw=1.3 * self.s, zorder=5.5)
        cx_, cy_ = x + ux * L, y + uy * L
        ang = math.degrees(math.atan2(-uy, ux))  # display y is flipped
        from matplotlib.patches import Wedge
        ax.add_patch(Wedge((cx_, cy_), 0.85 * size, ang - 90, ang + 90, fc=T["bg"], ec=T["ink2"], lw=1.3 * self.s, zorder=5.6))
        px, py = -uy, ux
        ax.plot([cx_ - px * 0.85 * size, cx_ + px * 0.85 * size], [cy_ - py * 0.85 * size, cy_ + py * 0.85 * size],
                color=T["ink2"], lw=1.3 * self.s, zorder=5.7)

    def _valve_ball(self, ax, x, y, open_, vertical, size, needle):
        """Ball-valve icon: round body on the tube, the bore drawn along the pipe when open, across when shut;
        a needle valve shows a tapered pin."""
        T = self.T
        r = 0.95 * size
        ax.add_patch(plt.Circle((x, y), r, fc=T["bg"], ec=T["ink"], lw=1.8 * self.s, zorder=6))
        if needle:
            ax.add_patch(Polygon([(x - 0.35 * r, y - 0.75 * r), (x + 0.35 * r, y - 0.75 * r), (x, y + 0.35 * r)],
                                 closed=True, fc=T["ink"], ec="none", zorder=7))
            ax.plot([x - 1.25 * r, x + 1.25 * r], [y + 1.25 * r, y - 1.25 * r], color=T["ink2"], lw=1.1 * self.s, zorder=7)
            ax.add_patch(Polygon([(x + 1.25 * r, y - 1.25 * r), (x + 0.8 * r, y - 1.15 * r), (x + 1.15 * r, y - 0.8 * r)],
                                 closed=True, fc=T["ink2"], ec="none", zorder=7))
            return
        along = (vertical and open_) or (not vertical and not open_)
        if along:
            ax.plot([x, x], [y - 0.72 * r, y + 0.72 * r], color=T["ink"] if open_ else T["muted"], lw=4.0 * self.s,
                    zorder=7, solid_capstyle="round")
        else:
            ax.plot([x - 0.72 * r, x + 0.72 * r], [y, y], color=T["ink"] if open_ else T["muted"], lw=4.0 * self.s,
                    zorder=7, solid_capstyle="round")

    def _valve3_ball(self, ax, x, y, dirs, open_ports, size):
        """3-way ball valve: the L bore joins the common port (the bed, dirs[0]) to the open port."""
        T = self.T
        r = 1.05 * size
        vec = {"u": (0, -1), "d": (0, 1), "l": (-1, 0), "r": (1, 0)}
        ax.add_patch(plt.Circle((x, y), r, fc=T["bg"], ec=T["ink"], lw=1.8 * self.s, zorder=6))
        for p in dirs:  # port stubs
            ux, uy = vec[p]
            ax.plot([x + ux * r, x + ux * 1.25 * r], [y + uy * r, y + uy * 1.25 * r], color=T["ink"], lw=1.8 * self.s, zorder=6)
        ports = [p for p in dirs if p in open_ports]
        if ports:
            for p in ports:
                ux, uy = vec[p]
                ax.plot([x, x + ux * 0.72 * r], [y, y + uy * 0.72 * r], color=T["ink"], lw=4.0 * self.s, zorder=7,
                        solid_capstyle="round")
        else:  # shut: the bore faces the blank side
            ax.plot([x - 0.4 * r, x + 0.4 * r], [y - 0.4 * r, y + 0.4 * r], color=T["muted"], lw=4.0 * self.s, zorder=7,
                    solid_capstyle="round")
        ax.add_patch(plt.Circle((x, y), 2.4, fc=T["ink"], ec="none", zorder=8))

    def _valve3(self, ax, x, y, dirs, open_ports, size=14):
        """3-way valve: one triangle per port (apex at the centre), filled where the path is open.
        dirs: port directions ('u', 'd', 'l', 'r'); open_ports: set of open ones (the common port fills
        whenever any other port is open)."""
        style = self.cfg.get("layout", {}).get("valve_style", "plain")
        if style == "ball":
            return self._valve3_ball(ax, x, y, dirs, open_ports, size)
        T = self.T
        s = size
        vec = {"u": (0, -1), "d": (0, 1), "l": (-1, 0), "r": (1, 0)}
        if style == "isa":
            out = {"r": "l", "l": "r"}[dirs[2]]  # actuator on the outer side, away from the inner line
            self._actuator(ax, x, y, out, size)
        for p in dirs:
            ux, uy = vec[p]
            px, py = -uy, ux
            tri = [(x, y), (x + ux * s + px * s * 0.78, y + uy * s + py * s * 0.78), (x + ux * s - px * s * 0.78, y + uy * s - py * s * 0.78)]
            on = p in open_ports
            ax.add_patch(Polygon(tri, closed=True, fc=T["ink"] if on else T["bg"], ec=T["ink"] if on else T["muted"],
                                 lw=1.6 * self.s, zorder=6))
        ax.add_patch(plt.Circle((x, y), 2.6, fc=T["ink"], ec="none", zorder=7))

    def _gauge(self, fig, ax, k, fr):
        """Live O2 analyzer dial on bed k's top outlet: log scale 10 ppm .. 25 %, spec band <= 0.5 %."""
        T = self.T
        gx, gy = self.L["gauges"][k]
        r = self.L["gauge_r"]
        y_o2 = float(fr[f"y{k}"][IO2, -1])
        lo, hi = math.log10(1e-5), math.log10(0.25)
        a0, a1 = 225.0, -45.0  # degrees, low at lower left, high at lower right

        def ang(v):
            f = (math.log10(min(max(v, 1e-5), 0.25)) - lo) / (hi - lo)
            return math.radians(a0 + (a1 - a0) * f)

        ax.add_patch(plt.Circle((gx, gy), r, fc=T["bg"], ec=T["metal"], lw=2.2 * self.s, zorder=8))
        # spec band: up to 0.5 % O2 in the N2 colour
        th = np.linspace(ang(1e-5), ang(5e-3), 40)
        ax.plot(gx + 0.80 * r * np.cos(th), gy - 0.80 * r * np.sin(th), color=T["N2"], lw=3.2 * self.s, zorder=9,
                solid_capstyle="butt")
        th = np.linspace(ang(5e-3), ang(0.25), 40)
        ax.plot(gx + 0.80 * r * np.cos(th), gy - 0.80 * r * np.sin(th), color=T["O2"], lw=3.2 * self.s, zorder=9,
                solid_capstyle="butt", alpha=0.55)
        for v in (1e-4, 1e-3, 1e-2, 1e-1):
            t = ang(v)
            ax.plot([gx + 0.62 * r * math.cos(t), gx + 0.92 * r * math.cos(t)],
                    [gy - 0.62 * r * math.sin(t), gy - 0.92 * r * math.sin(t)], color=T["ink2"], lw=1.0 * self.s, zorder=9)
        t = ang(y_o2)
        ax.plot([gx, gx + 0.78 * r * math.cos(t)], [gy, gy - 0.78 * r * math.sin(t)], color=T["ink"], lw=2.0 * self.s,
                zorder=10, solid_capstyle="round")
        ax.add_patch(plt.Circle((gx, gy), 3.2, fc=T["ink"], ec="none", zorder=11))
        val = f"{1e6 * y_o2:,.0f} ppm" if y_o2 < 0.01 else f"{100 * y_o2:.1f} %"
        self.text(fig, gx, gy + r + 5, "O$_2$ out", size=9.5, ha="center", color=T["muted"])
        self.text(fig, gx, gy + r + 19, val, size=12, ha="center", fontweight="bold")

    def _pid_threeway(self, fig, fr):
        T = self.T
        ax = fig.add_axes([0, 0, 1, 1], zorder=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_axis_off()
        net, g = self._net_threeway()
        net.draw_tubes(ax)
        net.draw_gas(ax, fr, [T[sp] for sp in SPECIES])
        net.draw_chevrons(ax, fr)
        cx = g["cx"]
        eqb, eqt = bool(fr["open0_eq_bot"]), bool(fr["open0_eq_top"])
        for k in (0, 1):
            inward = "r" if k == 0 else "l"
            ob = set()
            if fr[f"open{k}_feed"] or eqb:
                ob |= {"u", inward}
            if fr[f"open{k}_vent"]:
                ob |= {"u", "d"}
            self._valve3(ax, cx[k], g["vb"], ("u", "d", inward), ob)
            ot = set()
            if fr[f"open{k}_product"] or eqt:
                ot |= {"d", inward}
            if fr[f"open{k}_purge"]:
                ot |= {"d", "u"}
            self._valve3(ax, cx[k], g["vt"], ("u", "d", inward), ot)
        feed_open = bool(fr["open0_feed"]) or bool(fr["open1_feed"])
        prod_open = bool(fr["open0_product"]) or bool(fr["open1_product"])
        self._valve(ax, g["txf"], 0.5 * (g["vb"] + g["ve"]) + 6, feed_open)
        self._valve(ax, g["txp"], 0.5 * (g["vt"] + g["pu"]), prod_open)
        self._valve(ax, g["txq"], 0.5 * (g["rl"] + g["pu"]), True, size=11, needle=True)
        for x, y, lab, ha in ((cx[0] + 22, g["vt"] - 12, "product · top equalization", "left"),
                              (cx[0] + 22, g["pu"] - 12, "purge", "left"),
                              (cx[0] + 22, g["vb"] - 12, "feed · bottom equalization", "left"),
                              (cx[0] + 22, g["ve"] - 12, "exhaust", "left")):
            self.text(fig, x, y, lab, size=10, color=T["muted"], va="bottom", ha=ha)
        tx, fo, vo = g["tx"], g["fo"], g["vo"]
        self._compressor(ax, tx + 24, fo)
        self._vessel(ax, tx + 77, fo - 45, 26, 70, None)
        self._box_icon(ax, tx + 100, fo - 20, 52, 40, "dryer", small=True)
        self.text(fig, tx, fo + 34, "compressor, air receiver, dryer · 7.5 barg, dew point +3 °C (context only)",
                  size=10, color=T["muted"])
        self._box_icon(ax, g["sx"], vo - 15, 56, 30, "silencer", small=True)
        self.text(fig, g["sx"] + 64, vo - 9, "vent", size=11, color=T["muted"])
        rx0, ry0, rx1, ry1 = g["receiver"]
        self._vessel(ax, g["rcx"], ry0, rx1 - rx0, ry1 - ry0, "N$_2$ receiver")
        self._regulator(ax, g["rcx"], ry1 + 30, vertical=True)
        self.text(fig, g["rcx"] + 16, ry1 + 40, "N$_2$ product\n30 SLPM\n6 barg", size=10.5, linespacing=1.2)
        for k in (0, 1):
            self._gauge(fig, ax, k, fr)

    def _pid(self, fig, fr):
        if self.L.get("pid") == "threeway":
            return self._pid_threeway(fig, fr)
        if self.L.get("pid") == "manifold":
            return self._pid_manifold(fig, fr)
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
        style = self.cfg.get("layout", {}).get("valve_style", "plain")
        if style in ("isa", "ball"):
            self._valve(ax, x, y, True, vertical=vertical, size=10, actuator=False)
            # pressure regulator: diaphragm dome on a stem plus a spring (zigzag) above it
            side = 1
            if vertical:
                ax.plot([x, x + 16], [y, y], color=T["ink2"], lw=1.2 * self.s, zorder=6)
                from matplotlib.patches import Wedge
                ax.add_patch(Wedge((x + 16, y), 8, -90, 90, fc=T["bg"], ec=T["ink2"], lw=1.2 * self.s, zorder=6))
                ax.plot([x + 16, x + 16], [y - 8, y + 8], color=T["ink2"], lw=1.2 * self.s, zorder=6)
                zx = x + 24 + np.arange(7) * 3.0
                zy = y + np.array([0, -4, 4, -4, 4, -4, 0])
                ax.plot(zx, zy, color=T["ink2"], lw=1.0 * self.s, zorder=6)
            else:
                ax.plot([x, x], [y, y - 16], color=T["ink2"], lw=1.2 * self.s, zorder=6)
                from matplotlib.patches import Wedge
                ax.add_patch(Wedge((x, y - 16), 8, 0, 180, fc=T["bg"], ec=T["ink2"], lw=1.2 * self.s, zorder=6))
                zy = y - 24 - np.arange(7) * 3.0
                zx = x + np.array([0, -4, 4, -4, 4, -4, 0])
                ax.plot(zx, zy, color=T["ink2"], lw=1.0 * self.s, zorder=6)
            del side
            return
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

    def _species_panel(self, a, y, show_y):
        """All five species of one bed on one log mole-fraction axis (0.1 ppm .. 100 %)."""
        T = self.T
        for i in (IN2, IO2, IAR, ICO2, IH2O):
            a.plot(np.maximum(y[i], 1e-9), self.z, color=T[SPECIES[i]], lw=2.0 * self.s)
        a.axhspan(0, self.z_al, color=T["highlight"], lw=0, zorder=0)
        a.set(xscale="log", xlim=(1e-7, 1.6), ylim=(0, 1))
        a.set_xticks([1e-6, 1e-3, 1])
        a.set_xticklabels(["1 ppm", "0.1 %", "100 %"])
        a.set_xticks([1e-7, 1e-5, 1e-4, 1e-2, 1e-1], minor=True)
        a.tick_params(which="minor", length=0)
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.grid(True, which="minor", color=T["grid"], lw=0.6 * self.s)
        a.set_yticks([0, 0.5, 1] if show_y else [])
        if show_y:
            a.set_ylabel("height  z / L", fontsize=self.fs(11.5))
        a.set_xlabel("mole fraction in the gas (log)", fontsize=self.fs(11))

    def _species_key(self, fig, box, title=None):
        T = self.T
        x, y = box[0], box[1] - 30
        if title:
            self.text(fig, x, y - 2, title, size=12, fontweight="bold")
            x += 62
        for j, i in enumerate((IN2, IO2, IAR, ICO2, IH2O)):
            xx = x + j * 44
            fig.lines.append(plt.Line2D([self.fx(xx + 5)], [self.fy(y + 8)], marker="o", ms=6 * self.s,
                                        color=T[SPECIES[i]], mec=T["bg"], transform=fig.transFigure, figure=fig))
            self.text(fig, xx + 12, y, SPECIES_TEX[SPECIES[i]], size=10.5, color=T["ink2"])

    def _profiles(self, fig, fr):
        if self.L["prof"] == "per_bed_all":
            for k in (0, 1):
                box = self.L["prof_boxes"][k]
                a = self.box_axes(fig, box)
                self._species_panel(a, fr[f"y{k}"], show_y=True)
                self._axis_style(a)
                self._species_key(fig, box, title=f"Bed {k + 1}")
            return
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
