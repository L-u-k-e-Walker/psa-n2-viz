"""Physics cache for the renders: time map and the solver state at every frame (env viz).

* ``TimeMap`` maps video time to cycle time, piecewise linear per half-cycle, so the short
  steps (equalization, the start of production/blowdown) are stretched in video time while the
  on-screen clock shows the true t / t_cycle. The second half mirrors the first, so the loop is
  seamless.
* ``frame_cache`` integrates one plain cycle from the saved CSS state and samples it at every
  frame of the final (30 fps) timeline. Drafts take every second frame, so a draft is an exact
  subsample of its final. The cache is keyed by a hash of the CSS state file, the timing table
  and the frame rate, and stored in render/cache/ (gitignored): changing the look never re-runs
  the model.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from psan2viz import css  # noqa: E402
from psan2viz.analysis import Cycle  # noqa: E402

CSS_DIR = ROOT / "model" / "css" / "final"
FPS_TIMELINE = 30  # frame states are always computed on the final's timeline
CACHE_VERSION = 5  # bump when the cached fields change


class TimeMap:
    def __init__(self, t_half, cycle_knots, video_knots):
        self.t_half = float(t_half)
        self.ck = np.asarray(cycle_knots, float)
        self.vk = np.asarray(video_knots, float)
        if abs(self.ck[-1] - self.t_half) > 1e-9 or np.any(np.diff(self.ck) <= 0) or np.any(np.diff(self.vk) <= 0):
            raise ValueError("timing knots must increase and the cycle knots must end at t_half")
        self.v_half = float(self.vk[-1])
        self.video_s = 2 * self.v_half

    def cycle_time(self, tv):
        """Video time (s, wrapped to the loop) -> cycle time (s)."""
        tv = np.mod(np.asarray(tv, float), self.video_s)
        half = (tv >= self.v_half).astype(float)
        return half * self.t_half + np.interp(tv - half * self.v_half, self.vk, self.ck)

    def video_time(self, tc):
        tc = np.mod(np.asarray(tc, float), 2 * self.t_half)
        half = (tc >= self.t_half).astype(float)
        return half * self.v_half + np.interp(tc - half * self.t_half, self.ck, self.vk)


def file_hash(path, extra=""):
    h = hashlib.sha256(Path(path).read_bytes())
    h.update(extra.encode())
    return h.hexdigest()[:12]


def load_cycle(css_dir=CSS_DIR):
    case, pl, x = css.load(css_dir)
    _, recs = pl.run_cycle(x)
    return Cycle(pl, recs)


def frame_cache(timing, css_dir=CSS_DIR, cache_dir=HERE / "cache", log=print):
    """Per-frame solver data on the 30 fps timeline, cached. Returns (dict of arrays, TimeMap, Cycle)."""
    css_file = Path(css_dir) / "css_state.npz"
    key = file_hash(css_file, json.dumps(timing, sort_keys=True) + f"fps{FPS_TIMELINE}v{CACHE_VERSION}")
    path = Path(cache_dir) / f"frames_{key}.npz"
    cyc = load_cycle(css_dir)
    tm = TimeMap(cyc.case.t_half, timing["half_cycle_knots"], timing["half_video_knots"])
    if path.exists():
        d = dict(np.load(path, allow_pickle=False))
        return d, tm, cyc
    n = int(round(tm.video_s * FPS_TIMELINE))
    tv = np.arange(n) / FPS_TIMELINE
    tc = tm.cycle_time(tv)
    frames = [cyc.frame(t) for t in tc]
    keys_arr = [k for k, v in frames[0].items() if isinstance(v, np.ndarray)]
    keys_num = [k for k, v in frames[0].items() if isinstance(v, (float, int, bool, np.floating, np.bool_))]
    keys_str = [k for k, v in frames[0].items() if isinstance(v, str)]
    out = {k: np.stack([f[k] for f in frames]) for k in keys_arr}
    out.update({k: np.array([f[k] for f in frames]) for k in keys_str})
    out.update({k: np.array([f[k] for f in frames]) for k in keys_num})
    out["t_video"] = tv
    out["t_cycle"] = tc
    # full-cycle traces for the time-series panels (fine sampling of the true cycle)
    ts, fr = cyc.sample(150)
    sp = list(cyc.case.species)
    out["trace_t"] = ts
    out["trace_P0"] = np.array([f["P0"] for f in fr])
    out["trace_P1"] = np.array([f["P1"] for f in fr])
    out["trace_PT"] = np.array([f["p_tank"] for f in fr])
    out["trace_yT"] = np.stack([f["y_tank"] for f in fr])
    out["trace_y0_top"] = np.stack([f["y0"][:, -1] for f in fr])
    out["trace_y1_top"] = np.stack([f["y1"][:, -1] for f in fr])
    # feed flow over the cycle for the compressor-power chart, dense at the start of each step (the
    # pressurization spike lasts well under a second)
    tp = np.concatenate([cyc.starts[i] + d_ * np.linspace(0.0, 1.0, 300, endpoint=False) ** 2.5
                         for i, d_ in enumerate(cyc.dur)] + [[cyc.T]])
    frp = [cyc.frame(t) for t in tp]
    out["power_t"] = tp
    out["power_Ffeed"] = np.array([f["F0_feed"] + f["F1_feed"] for f in frp])
    out["z"] = cyc.z
    out["z_faces"] = cyc.pl.g.z_faces / cyc.pl.g.length
    out["z_al"] = np.array(cyc.z_al)
    out["species"] = np.array(sp)
    out["cache_key"] = np.array(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    log(f"frame cache written: {path.name} ({n} frames)")
    return dict(np.load(path, allow_pickle=False)), tm, cyc
