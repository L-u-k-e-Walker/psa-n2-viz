"""Gate 0: which OVITO renderer works headless here, and how fast (env viz).

Renders the same small test scene (packed spheres + a vertex-coloured mesh, the
kind of content the bed scene uses) with each candidate renderer, each in its
own subprocess (a failing OpenGL context must not take the others down).
Writes docs/gate0/render_<name>.png and docs/gate0/gate0.json.

Run from the repo root: conda run -n viz python -I tools/gate0_render_test.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "gate0"
CANDIDATES = {
    "opengl": "OpenGLRenderer()",
    "tachyon_draft": "TachyonRenderer(ambient_occlusion=False, antialiasing=False, shadows=False)",
    "tachyon_final": "TachyonRenderer(ambient_occlusion=True, ambient_occlusion_brightness=0.85, "
                     "ambient_occlusion_samples=12, antialiasing_samples=12, shadows=False)",
}

CHILD = r'''
import warnings, time, sys
warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
import numpy as np
from ovito.data import DataCollection
from ovito.pipeline import Pipeline, StaticSource
from ovito.vis import Viewport, TachyonRenderer, OpenGLRenderer, TriangleMeshVis, ParticlesVis
rng = np.random.default_rng(1)
n = 6000
pos = np.c_[rng.uniform(-7, 7, n), rng.uniform(0, 7, n), rng.uniform(0, 40, n)]
dc = DataCollection()
p = dc.create_particles(count=n)
p.create_property("Position", data=pos)
p.create_property("Radius", data=np.full(n, 0.49))
p.create_property("Color", data=np.c_[np.linspace(0.3, 0.9, n), np.full(n, 0.45), np.linspace(0.8, 0.2, n)])
p.vis = ParticlesVis(radius=0.5)
m = dc.triangle_meshes.create("wall")
m.set_vertices(np.array([[-8, 8, -1], [8, 8, -1], [8, 8, 41], [-8, 8, 41]], float))
m.set_faces(np.array([[0, 1, 2], [0, 2, 3]]))
m.set_vertex_colors(np.array([[0.7, 0.75, 0.8, 1]] * 4, float))
m.vis = TriangleMeshVis(transparency=0.6, highlight_edges=False)
pipe = Pipeline(source=StaticSource(data=dc)); pipe.add_to_scene()
vp = Viewport(type=Viewport.Type.Ortho, camera_pos=(0, -80, 20), camera_dir=(0, 1, 0), fov=22.0)
r = RENDERER
t0 = time.time()
vp.render_image(size=(960, 540), filename=sys.argv[1], background=(0.1, 0.1, 0.1), alpha=False, renderer=r)
t1 = time.time()
vp.render_image(size=(960, 540), filename=sys.argv[1], background=(0.1, 0.1, 0.1), alpha=False, renderer=r)
print("TIMES", t1 - t0, time.time() - t1)
'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, ctor in CANDIDATES.items():
        png = OUT / f"render_{name}.png"
        png.unlink(missing_ok=True)
        t0 = time.time()
        pr = subprocess.run([sys.executable, "-I", "-c", CHILD.replace("RENDERER", ctor), str(png)],
                            capture_output=True, text=True, timeout=600)
        wall = time.time() - t0
        times = [ln for ln in pr.stdout.splitlines() if ln.startswith("TIMES")]
        ok = pr.returncode == 0 and png.exists() and png.stat().st_size > 1000
        entry = dict(ok=ok, returncode=pr.returncode, wall_s=round(wall, 2))
        if times:
            first, second = map(float, times[0].split()[1:])
            entry.update(first_frame_s=round(first, 3), second_frame_s=round(second, 3))
        if not ok:
            entry["stderr_tail"] = pr.stderr.strip().splitlines()[-6:]
        res[name] = entry
        print(name, json.dumps(entry), flush=True)
    if res["opengl"]["ok"]:
        res["draft_renderer"] = "opengl"
    elif res["tachyon_draft"]["ok"]:
        res["draft_renderer"] = "tachyon_draft"
    else:
        res["draft_renderer"] = None
    res["pass"] = res["draft_renderer"] is not None
    (OUT / "gate0.json").write_text(json.dumps(res, indent=1))
    print("draft renderer:", res["draft_renderer"])


if __name__ == "__main__":
    main()
