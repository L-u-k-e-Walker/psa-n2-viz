"""One renderer, three quality tiers (env viz). Every look setting is in render/config.toml.

usage:
  python render.py --mock [--layout A|B|C|all] [--theme dark]   contact sheet of 960x540 stills (matplotlib only)
  python render.py --draft [--workers 8]                       full loop, 960x540, 15 fps, one theme, MP4 + sheet
  python render.py --final vNN [--workers 4]                   approved version's config: 1920x1080, 30 fps, both themes
  options: --css DIR (CSS state folder, default model/css/final), --version vNN (default: next free)

Mocks and drafts go to previews/vNN/ with config_<hash8>.toml (the exact config used). After each
mock or draft: stop and review (docs/REVIEWS.md). --final refuses to run unless docs/REVIEWS.md
marks the version "approved: final vNN".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]
from frames import CSS_DIR, frame_cache  # noqa: E402

PREVIEWS = ROOT / "previews"
REVIEWS = ROOT / "docs" / "REVIEWS.md"


def config_snapshot(cfg_path, folder):
    raw = Path(cfg_path).read_bytes()
    h = hashlib.sha256(raw).hexdigest()[:8]
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / f"config_{h}.toml"
    if not dst.exists():
        dst.write_bytes(raw)
    return h, dst


def next_version():
    nums = [int(m.group(1)) for p in PREVIEWS.glob("v[0-9][0-9]") if (m := re.match(r"v(\d\d)$", p.name))]
    return f"v{(max(nums) + 1) if nums else 1:02d}"


def frame_index_for(cycle_times, d):
    """Nearest frame (on the 30 fps timeline) to each requested cycle time."""
    tc = d["t_cycle"]
    return [int(np.argmin(np.abs(tc - t))) for t in cycle_times]


def contact_sheet(paths, out, cols, labels, theme_bg):
    from PIL import Image, ImageDraw
    ims = [Image.open(p).convert("RGB") for p in paths]
    w, h = ims[0].size
    rows = int(np.ceil(len(ims) / cols))
    pad, lab_h = 8, 26
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * (h + lab_h) + (rows + 1) * pad), theme_bg)
    dr = ImageDraw.Draw(sheet)
    for i, (im, lab) in enumerate(zip(ims, labels)):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + lab_h + pad)
        dr.text((x + 4, y + 4), lab, fill=(150, 150, 150))
        sheet.paste(im, (x, y + lab_h))
    sheet.save(out)


def load_metrics(css_dir):
    return json.loads((Path(css_dir) / "metrics.json").read_text())


def run_mock(args, cfg, cfg_path):
    from compose import LAYOUTS, Composer
    from psan2viz.style import THEMES, hex_rgb
    t0 = time.time()
    ver = args.version or next_version()
    folder = PREVIEWS / ver
    h, snap = config_snapshot(cfg_path, folder)
    d, tm, cyc = frame_cache(cfg["timing"], css_dir=args.css)
    met = load_metrics(args.css)
    tier = cfg["tiers"]["mock"]
    layouts = sorted(LAYOUTS) if args.layout == "all" else [args.layout or cfg["layout"]["option"]]
    idx = frame_index_for(tier["times_s"], d)
    sheets = []
    for lay in layouts:
        comp = Composer(args.theme, lay, d, met, cyc, size=tuple(tier["size"]), cfg=cfg)
        paths, labels = [], []
        for j, f in enumerate(idx):
            fr = {k: (v[f] if (isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(d["t_video"])) else v)
                  for k, v in d.items()}
            p = folder / f"mock_{lay}_{j:02d}.png"
            comp.draw(fr, p)
            paths.append(p)
            labels.append(f"layout {lay}   t = {fr['t_cycle']:.1f} s   (video {fr['t_video']:.2f} s)")
        out = folder / f"contact_{ver}_mock_{lay}.png"
        contact_sheet(paths, out, tier["sheet_cols"], labels, tuple(int(255 * v) for v in hex_rgb(THEMES[args.theme]["bg"])))
        sheets.append(out)
    wall = time.time() - t0
    (folder / "render_info.json").write_text(json.dumps(dict(tier="mock", version=ver, config_hash=h, layouts=layouts,
                                                             theme=args.theme, wall_s=round(wall, 1),
                                                             sheets=[str(s.relative_to(ROOT)) for s in sheets]), indent=1))
    print(f"mock {ver} ({', '.join(layouts)}) in {wall:.1f} s -> " + ", ".join(str(s.relative_to(ROOT)) for s in sheets))


def frame_dict(d, f):
    n = len(d["t_video"])
    return {k: (v[f] if (isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n) else v) for k, v in d.items()}


class BedRenderer:
    """Both 3D beds of one frame: one OVITO scene rendered twice (bed 2 mirrored), placed in the layout's boxes."""

    def __init__(self, theme, comp, d, cyc, mol, renderer, ao, aa_samples=12, ao_samples=12):
        from scene import BedScene
        from psan2viz.style import SPECIES, THEMES, hex_rgb
        sc = comp.size[0] / 1920
        self.boxes = [tuple(v * sc for v in b) for b in comp.L["beds"]]
        self.scene = BedScene(theme, self.boxes[0], float(d["z_al"]), renderer=renderer, ao=ao, aa_samples=aa_samples,
                              ao_samples=ao_samples)
        self.origins = [self.scene.origin_for(b) for b in self.boxes]
        self.colors = np.array([hex_rgb(THEMES[theme][s]) for s in SPECIES])
        self.q_ref = cyc.pl.q_scale
        self.z = d["z"]
        self.mol = mol
        # nozzle ends for the P&ID, in design px
        top, bot = self.scene.nozzle_px
        comp.nozzles = [(top / sc, bot / sc)] * 2

    def render(self, fr, f, tmp_stem):
        import matplotlib.pyplot as plt
        out = []
        for k in (0, 1):
            q = fr[f"q{k}"]
            o0, o1 = self.mol[f"off{k}"][f], self.mol[f"off{k}"][f + 1]
            dots = dict(pos=self.mol[f"pos{k}"][o0:o1].astype(float), sp=self.mol[f"sp{k}"][o0:o1].astype(int),
                        r=float(self.mol["r_dot"]))
            self.scene.set_state(q[1] / self.q_ref[1], q[4] / self.q_ref[4], self.z, mirror=(k == 1), dots=dots,
                                 colors=self.colors)
            tmp = Path(f"{tmp_stem}_bed{k}.png")
            self.scene.render(tmp)
            out.append((plt.imread(tmp), self.origins[k]))
            tmp.unlink()
        return out


def draft_setup(cfg, css_dir):
    from molecules import molecule_history
    from scene import Packing
    from psan2viz.style import SPECIES
    d, tm, cyc = frame_cache(cfg["timing"], css_dir=css_dir)
    met = load_metrics(css_dir)
    w = [cfg["molecules"]["weights"][s] for s in SPECIES]
    mol = molecule_history(d, Packing(), cyc.pl.g.eps, w, cfg["molecules"]["n_ref"], key_extra=str(d["cache_key"]))
    return d, tm, cyc, met, mol


def draft_worker(args, cfg, a, b, outdir, renderer, ver):
    from compose import Composer
    d, tm, cyc, met, mol = draft_setup(cfg, args.css)
    tier = cfg["tiers"]["draft"]
    comp = Composer(tier["theme"], cfg["layout"]["option"], d, met, cyc, size=tuple(tier["size"]), cfg=cfg)
    beds = BedRenderer(tier["theme"], comp, d, cyc, mol, renderer, ao=False)
    step = 30 // tier["fps"]
    for i in range(a, b):
        out = outdir / f"f_{i:04d}.png"
        if out.exists():
            continue
        f = i * step
        fr = frame_dict(d, f)
        imgs = beds.render(fr, f, str(outdir / f"_tmp_{i:04d}"))
        comp.draw(fr, out, beds_3d=imgs, watermark=f"DRAFT {ver}")


def time_renderers(args, cfg):
    """Seconds per frame of each draft renderer on the real scene (frame 0)."""
    import time as _t
    from compose import Composer
    d, tm, cyc, met, mol = draft_setup(cfg, args.css)
    tier = cfg["tiers"]["draft"]
    res = {}
    for name in ("opengl", "tachyon_fast"):
        comp = Composer(tier["theme"], cfg["layout"]["option"], d, met, cyc, size=tuple(tier["size"]), cfg=cfg)
        beds = BedRenderer(tier["theme"], comp, d, cyc, mol, "opengl" if name == "opengl" else "tachyon", ao=False)
        fr = frame_dict(d, 0)
        beds.render(fr, 0, "/tmp/_warm")  # first render pays the start-up
        t0 = _t.time()
        for f in (0, 60, 120):
            beds.render(frame_dict(d, f), f, "/tmp/_time")
        res[name] = (_t.time() - t0) / 3
        beds.scene.close()
    return res


def run_draft(args, cfg, cfg_path):
    import os
    import subprocess
    from psan2viz.style import THEMES, hex_rgb
    t0 = time.time()
    ver = args.version or next_version()
    folder = PREVIEWS / ver
    h, snap = config_snapshot(cfg_path, folder)
    tier = cfg["tiers"]["draft"]
    draft_setup(cfg, args.css)  # builds the caches once, before the workers start
    renderer = tier["renderer"]
    timing = None
    if renderer == "auto":
        timing = time_renderers(args, cfg)
        renderer = "opengl" if timing["opengl"] <= timing["tachyon_fast"] else "tachyon"
    outdir = ROOT / "frames" / f"{h}_draft_{tier['theme']}"
    outdir.mkdir(parents=True, exist_ok=True)
    n_frames = int(round(cfg["timing"]["half_video_knots"][-1] * 2 * tier["fps"]))
    edges = np.linspace(0, n_frames, args.workers + 1).astype(int)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    t1 = time.time()
    procs = [subprocess.Popen([sys.executable, "-I", __file__, "--draft-worker", str(a_), str(b_), str(outdir), renderer,
                               ver, "--config", str(snap), "--css", args.css], env=env)
             for a_, b_ in zip(edges[:-1], edges[1:])]
    codes = [p_.wait() for p_ in procs]
    t_frames = time.time() - t1
    done = len(list(outdir.glob("f_*.png")))
    mp4 = folder / f"draft_{ver}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(tier["fps"]), "-i", str(outdir / "f_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(mp4)], check=True)
    picks = np.linspace(0, n_frames - 1, tier["sheet_frames"]).astype(int)
    contact_sheet([outdir / f"f_{i:04d}.png" for i in picks], folder / f"contact_{ver}_draft.png", 2,
                  [f"frame {i}  (video {i / tier['fps']:.2f} s)" for i in picks],
                  tuple(int(255 * v) for v in hex_rgb(THEMES[tier["theme"]]["bg"])))
    wall = time.time() - t0
    info = dict(tier="draft", version=ver, config_hash=h, renderer=renderer, renderer_timing_s_per_frame=timing,
                frames=done, expected=n_frames, worker_exit=codes, wall_frames_s=round(t_frames, 1), wall_s=round(wall, 1),
                mp4=str(mp4.relative_to(ROOT)))
    (folder / "render_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def final_worker(args, cfg, a, b, outdir, theme):
    from compose import Composer
    d, tm, cyc, met, mol = draft_setup(cfg, args.css)
    tier = cfg["tiers"]["final"]
    comp = Composer(theme, cfg["layout"]["option"], d, met, cyc, size=tuple(tier["size"]), cfg=cfg)
    beds = BedRenderer(theme, comp, d, cyc, mol, "tachyon", ao=True, aa_samples=tier["aa_samples"],
                       ao_samples=tier["ao_samples"])
    step = 30 // tier["fps"]
    for i in range(a, b):
        out = outdir / f"f_{i:04d}.png"
        if out.exists():
            continue
        f = i * step
        fr = frame_dict(d, f)
        imgs = beds.render(fr, f, str(outdir / f"_tmp_{i:04d}"))
        comp.draw(fr, out, beds_3d=imgs)


def assemble_final(frames, mp4, gif, fps, gif_fps, gif_max_mb):
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "f_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-movflags", "+faststart",
                    str(mp4)], check=True)
    note = None
    for width in (1920, 1600, 1280, 960):
        vf = (f"fps={gif_fps},scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:"
              f"stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "f_%04d.png"),
                        "-vf", vf, str(gif)], check=True)
        mb = gif.stat().st_size / 1e6
        if mb <= gif_max_mb:
            if width != 1920:
                note = f"GIF downscaled to {width} px wide ({mb:.1f} MB; 1920 px would exceed {gif_max_mb} MB)"
            break
    return note


def run_final(args, cfg, cfg_path):
    import os
    import subprocess
    t0 = time.time()
    ver = args.final
    tier = cfg["tiers"]["final"]
    h = Path(cfg_path).stem.split("_")[-1]
    out_root = ROOT / "final"
    out_root.mkdir(exist_ok=True)
    draft_setup(cfg, args.css)
    n_frames = int(round(cfg["timing"]["half_video_knots"][-1] * 2 * tier["fps"]))
    info = dict(tier="final", version=ver, config_hash=h, themes={})
    for theme in tier["themes"]:
        outdir = ROOT / "frames" / f"{h}_final_{theme}"
        outdir.mkdir(parents=True, exist_ok=True)
        edges = np.linspace(0, n_frames, args.workers + 1).astype(int)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        t1 = time.time()
        procs = [subprocess.Popen([sys.executable, "-I", __file__, "--final-worker", str(a_), str(b_), str(outdir), theme,
                                   "--config", str(cfg_path), "--css", args.css], env=env)
                 for a_, b_ in zip(edges[:-1], edges[1:])]
        codes = [p_.wait() for p_ in procs]
        done = len(list(outdir.glob("f_*.png")))
        entry = dict(frames=done, expected=n_frames, worker_exit=codes, wall_frames_s=round(time.time() - t1, 1))
        if done == n_frames:
            stem = f"psa_n2_{theme}"
            entry["gif_note"] = assemble_final(outdir, out_root / f"{stem}.mp4", out_root / f"{stem}.gif", tier["fps"],
                                               tier["gif_fps"], tier["gif_max_mb"])
        info["themes"][theme] = entry
        print(theme, json.dumps(entry), flush=True)
    shutil.copy(cfg_path, out_root / Path(cfg_path).name)
    info["wall_s"] = round(time.time() - t0, 1)
    (out_root / "render_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def approved(ver):
    txt = REVIEWS.read_text() if REVIEWS.exists() else ""
    return re.search(rf"^\|\s*{ver}\s*\|.*approved: final {ver}", txt, re.M) is not None


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--draft", action="store_true")
    g.add_argument("--final", metavar="vNN")
    ap.add_argument("--layout", default=None)
    ap.add_argument("--theme", default="dark")
    ap.add_argument("--css", default=str(CSS_DIR))
    ap.add_argument("--version", default=None)
    ap.add_argument("--config", default=str(HERE / "config.toml"))
    ap.add_argument("--workers", type=int, default=8)
    if "--final-worker" in sys.argv:  # internal: one worker of a final render
        i = sys.argv.index("--final-worker")
        a_, b_, outdir, theme = sys.argv[i + 1:i + 5]
        rest = sys.argv[1:i] + sys.argv[i + 5:]
        ap2 = argparse.ArgumentParser()
        ap2.add_argument("--config")
        ap2.add_argument("--css", default=str(CSS_DIR))
        wa = ap2.parse_args(rest)
        cfg = tomllib.loads(Path(wa.config).read_text())
        final_worker(wa, cfg, int(a_), int(b_), Path(outdir), theme)
        return
    if "--draft-worker" in sys.argv:  # internal: one worker of a draft render
        i = sys.argv.index("--draft-worker")
        a_, b_, outdir, renderer, ver = sys.argv[i + 1:i + 6]
        rest = sys.argv[1:i] + sys.argv[i + 6:]
        ap2 = argparse.ArgumentParser()
        ap2.add_argument("--config")
        ap2.add_argument("--css", default=str(CSS_DIR))
        wa = ap2.parse_args(rest)
        cfg = tomllib.loads(Path(wa.config).read_text())
        draft_worker(wa, cfg, int(a_), int(b_), Path(outdir), renderer, ver)
        return
    args = ap.parse_args()
    if args.final:
        if not approved(args.final):
            sys.exit(f"--final {args.final}: docs/REVIEWS.md does not mark {args.final} 'approved: final {args.final}'")
        snaps = sorted((PREVIEWS / args.final).glob("config_*.toml"))
        if len(snaps) != 1:
            sys.exit(f"--final {args.final}: expected one config snapshot in previews/{args.final}, found {len(snaps)}")
        args.config = str(snaps[0])
    cfg = tomllib.loads(Path(args.config).read_text())
    if args.mock:
        run_mock(args, cfg, args.config)
    elif args.draft:
        run_draft(args, cfg, args.config)
    else:
        run_final(args, cfg, args.config)


if __name__ == "__main__":
    main()
