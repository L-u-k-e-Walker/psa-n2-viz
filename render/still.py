"""One frame at final quality (Tachyon with ambient occlusion) for look checks (env viz). Not a final render:
no MP4/GIF and no approval needed; for judging the look before a draft or final.

usage: python render/still.py OUT.png --t 20 [--css model/css/final] [--quality final|draft]
"""
import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
from render import BedRenderer, draft_setup, frame_dict  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--t", type=float, default=20.0, help="cycle time (s)")
ap.add_argument("--css", default=str(HERE.parent / "model" / "css" / "final"))
ap.add_argument("--quality", default="final", choices=["final", "draft"])
ap.add_argument("--config", default=str(HERE / "config.toml"))
a = ap.parse_args()
cfg = tomllib.loads(Path(a.config).read_text())
from compose import Composer  # noqa: E402

d, tm, cyc, met, mol = draft_setup(cfg, a.css)
tier = cfg["tiers"]["final"]
theme = tier["themes"][0]
comp = Composer(theme, cfg["layout"]["option"], d, met, cyc, size=tuple(tier["size"]), cfg=cfg)
if a.quality == "final":
    beds = BedRenderer(theme, comp, d, cyc, mol, "tachyon", ao=True, aa_samples=tier["aa_samples"], ao_samples=tier["ao_samples"])
else:
    beds = BedRenderer(theme, comp, d, cyc, mol, "opengl", ao=False)
f = int(np.argmin(np.abs(d["t_cycle"] - a.t)))
fr = frame_dict(d, f)
imgs = beds.render(fr, f, str(Path(a.out).with_suffix("")) + "_tmp")
comp.draw(fr, a.out, beds_3d=imgs)
print(a.out, f"t = {d['t_cycle'][f]:.2f} s")
