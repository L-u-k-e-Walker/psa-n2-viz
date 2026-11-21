"""OVITO scene of one packed bed, rendered into its layout box (env viz).

Origin: psa-viz column/render/scene.py (BedScene, cylinder_mesh; commit 709af6f): randomly
packed spheres in a cut-away column, coloured from the solver, back half of a glass wall,
caps, distributor plates and nozzles, orthographic camera slightly from above. Changed: one
scene renders either bed (bed 2 is the mirrored packing), the camera scale and image size are
set so the packed height fills the layout's bed box exactly, the alumina layer has its own ramp,
and the renderer is chosen by tier (OpenGL or Tachyon without AO for drafts, Tachyon with AO
for finals). OVITO lessons kept: RGBA vertex colours, set_normals (3F)x3, a fresh clone of the
source data per frame, pipelines removed before another theme is built in the same process.
Geometry units: particle diameter = 1 (packing.npz).
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from psan2viz.style import THEMES, apply_ramp, hex_rgb, loading_ramp  # noqa: E402

ELEV = np.radians(14.0)
PLENUM = 1.4  # gap between the packed bed and the caps (particle diameters)
CAP = 0.5
NOZZLE = 1.6


class Packing:
    def __init__(self):
        d = np.load(HERE / "packing.npz")
        self.pos, self.r = d["pos"], float(d["radius"])
        self.R, self.H = float(d["R_col"]), float(d["H_bed"])


def cylinder_mesh(R, z0, z1, th0, th1, n=96, cap=False):
    th = np.linspace(th0, th1, n)
    v = np.concatenate([np.c_[R * np.cos(th), R * np.sin(th), np.full(n, z0)],
                        np.c_[R * np.cos(th), R * np.sin(th), np.full(n, z1)]])
    f = []
    for i in range(n - 1):
        f += [[i, i + 1, n + i + 1], [i, n + i + 1, n + i]]
    nrm = np.concatenate([np.c_[np.cos(th), np.sin(th), np.zeros(n)]] * 2)
    if cap:
        for z in (z0, z1):
            c = len(v)
            v = np.concatenate([v, [[0, 0, z]], np.c_[R * np.cos(th), R * np.sin(th), np.full(n, z)]])
            nrm = np.concatenate([nrm, np.tile([0, 0, 1 if z == z1 else -1], (n + 1, 1))])
            for i in range(n - 1):
                f.append([c, c + 1 + i, c + 2 + i])
    return np.asarray(v, float), np.asarray(f, int), np.asarray(nrm, float)


class BedScene:
    """One bed. ``box`` = (x0, y0, x1, y1) in output pixels: the packed bed's projected extent."""

    def __init__(self, theme, box, z_al, renderer="tachyon", ao=True, aa_samples=12, ao_samples=12, cut_y=0.0):
        from ovito.data import DataCollection
        from ovito.pipeline import Pipeline, StaticSource
        from ovito.vis import OpenGLRenderer, ParticlesVis, TachyonRenderer, TriangleMeshVis, Viewport

        self.theme, self.T = theme, THEMES[theme]
        self.pack = Packing()
        p = self.pack
        self.keep = p.pos[:, 1] > cut_y
        self.base = p.pos[self.keep]
        rng = np.random.default_rng(11)
        self.jitter = 1.0 + rng.normal(0, 0.035, len(self.base))[:, None]
        self.zrel = self.base[:, 2] / p.H
        self.is_al = self.zrel < z_al
        self.ramp_cms = loading_ramp(theme, "O2", deep=True)
        self.ramp_al = loading_ramp(theme, "H2O", deep=True, bare_key="bare_alumina")
        # pixel geometry: scale so that the packed height fills the box height
        x0, y0, x1, y1 = box
        self.box = box
        self.scale = (y1 - y0) / (p.H * np.cos(ELEV))  # px per particle diameter
        R = p.R + 0.08
        top = p.H + PLENUM + CAP + NOZZLE
        margin_top = (PLENUM + CAP + NOZZLE) * np.cos(ELEV) + R * np.sin(ELEV) + 1.0
        margin_bot = (PLENUM + CAP + NOZZLE) * np.cos(ELEV) + R * np.sin(ELEV) + 1.0
        self.w = int(np.ceil((2 * (R + 0.6)) * self.scale)) + 4
        self.h = int(np.ceil((p.H * np.cos(ELEV) + margin_top + margin_bot) * self.scale)) + 4
        self.origin = (0.5 * (x0 + x1) - self.w / 2, y0 - margin_top * self.scale - 2)  # image top-left in the frame
        self.data = DataCollection()
        pt = self.data.create_particles(count=len(self.base))
        pt.create_property("Position", data=self.base)
        pt.create_property("Radius", data=np.full(len(self.base), p.r * 0.985))
        pt.create_property("Color", data=np.tile(hex_rgb(self.T["bare"]), (len(self.base), 1)))
        pt.vis = ParticlesVis(radius=0.5)
        z0, z1 = -PLENUM, p.H + PLENUM
        v, f, n = cylinder_mesh(R, z0, z1, 0.0, np.pi)
        m = self.data.triangle_meshes.create("wall")
        m.set_vertices(v)
        m.set_faces(f)
        m.set_normals(n[f].reshape(-1, 3))
        m.vis = TriangleMeshVis(color=tuple(hex_rgb(self.T["glass"])), transparency=0.72, highlight_edges=False,
                                backface_culling=False)
        metal = tuple(hex_rgb(self.T["metal"]))
        for name, (zz0, zz1, rr, th0, th1, cap) in {
            "cap_bot": (z0 - CAP, z0, R + 0.35, 0.0, np.pi, True),
            "cap_top": (z1, z1 + CAP, R + 0.35, 0.0, np.pi, True),
            "noz_bot": (z0 - CAP - NOZZLE, z0 - CAP, 1.0, 0.0, 2 * np.pi, True),
            "noz_top": (z1 + CAP, z1 + CAP + NOZZLE, 1.0, 0.0, 2 * np.pi, True),
        }.items():
            v, f, n = cylinder_mesh(rr, zz0, zz1, th0, th1, cap=cap)
            mm = self.data.triangle_meshes.create(name)
            mm.set_vertices(v)
            mm.set_faces(f)
            mm.vis = TriangleMeshVis(color=metal, transparency=0.0, highlight_edges=False)
        self.pipe = Pipeline(source=StaticSource(data=self.data))
        self.pipe.add_to_scene()
        self.gas = DataCollection()
        self.gas_pipe = Pipeline(source=StaticSource(data=self.gas))
        self.gas_pipe.add_to_scene()
        # camera: orthographic, from slightly above, looking at the cut face (-y side removed)
        d = np.array([0.0, np.cos(ELEV), -np.sin(ELEV)])
        self.dir = d
        self.up = np.array([0.0, np.sin(ELEV), np.cos(ELEV)])
        self.target = np.array([0.0, 0.0, p.H / 2 + (margin_bot - margin_top) / (2 * np.cos(ELEV))])
        self.fov = self.h / (2 * self.scale)
        self.vp = Viewport(type=Viewport.Type.Ortho, camera_pos=tuple(self.target - 80 * d), camera_dir=tuple(d),
                           fov=self.fov)
        if renderer == "opengl":
            self.renderer = OpenGLRenderer(antialiasing_level=2)
        elif ao:
            self.renderer = TachyonRenderer(ambient_occlusion=True, ambient_occlusion_brightness=0.85,
                                            ambient_occlusion_samples=ao_samples, antialiasing_samples=aa_samples,
                                            direct_light_intensity=0.95, shadows=False)
        else:
            self.renderer = TachyonRenderer(ambient_occlusion=False, antialiasing=False, direct_light_intensity=0.95,
                                            shadows=False)
        self.nozzle_px = (self.project([0, -1.0, top])[1], self.project([0, -1.0, -PLENUM - CAP - NOZZLE])[1])

    def origin_for(self, box):
        """Top-left pixel of this scene's image when its bed fills ``box`` (same size as the scene's box)."""
        x0, y0, x1, y1 = box
        dx = 0.5 * (x0 + x1) - 0.5 * (self.box[0] + self.box[2])
        dy = y0 - self.box[1]
        return self.origin[0] + dx, self.origin[1] + dy

    def project(self, p):
        """World point -> pixel (x, y from the top) in the full frame."""
        s = self.scale
        rgt = np.cross(self.dir, self.up)
        q = np.asarray(p, float) - self.target
        return self.origin[0] + self.w / 2 + s * (q @ rgt), self.origin[1] + self.h / 2 - s * (q @ self.up)

    def set_state(self, q_cms_rel, q_al_rel, z_cells, mirror=False, dots=None, colors=None):
        """Colour particles from the solver's loading profiles (relative to feed equilibrium) at their height."""
        from ovito.data import DataCollection
        v_cms = np.interp(self.zrel, z_cells, q_cms_rel)
        v_al = np.interp(self.zrel, z_cells, q_al_rel)
        col = np.where(self.is_al[:, None], apply_ramp(self.ramp_al, v_al), apply_ramp(self.ramp_cms, v_cms))
        col = np.clip(col * self.jitter, 0, 1)
        dc = self.data.clone()
        pos = self.base.copy()
        if mirror:
            pos[:, 0] = -pos[:, 0]
        dc.particles_.create_property("Position", data=pos)
        dc.particles_.create_property("Color", data=col)
        self.pipe.source.data = dc
        g = DataCollection()
        if dots is not None and len(dots["pos"]):
            pt = g.create_particles(count=len(dots["pos"]))
            pp = dots["pos"].copy()
            if mirror:
                pp[:, 0] = -pp[:, 0]
            pt.create_property("Position", data=pp)
            pt.create_property("Radius", data=np.full(len(pp), dots["r"]))
            pt.create_property("Color", data=colors[dots["sp"]])
        self.gas_pipe.source.data = g

    def close(self):
        self.pipe.remove_from_scene()
        self.gas_pipe.remove_from_scene()

    def render(self, path):
        self.vp.render_image(size=(self.w, self.h), filename=str(path), background=hex_rgb(self.T["bg"]), alpha=True,
                             renderer=self.renderer)
