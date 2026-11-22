"""Gate 4: check the final MP4/GIF outputs with ffprobe (env viz). Writes docs/gate4_qc.json.

Origin: psa-viz tools/gate4_qc.py (commit 709af6f), adapted to this project's outputs (final/,
frames/<hash>_final_<theme>/) and frame count from the approved config.
Run from the repo root: conda run -n viz python -I tools/gate4_qc.py
"""
import json
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "final"
info = json.loads((FINAL / "render_info.json").read_text())
cfg = tomllib.loads(next(FINAL.glob("config_*.toml")).read_text())
tier = cfg["tiers"]["final"]
video_s = 2 * cfg["timing"]["half_video_knots"][-1]
n_frames = int(round(video_s * tier["fps"]))


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,nb_read_frames,codec_name,pix_fmt,r_frame_rate:format=duration",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    s = d["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return dict(width=s["width"], height=s["height"], frames=int(s["nb_read_frames"]), codec=s["codec_name"],
                pix_fmt=s.get("pix_fmt"), fps=num / den, duration_s=float(d["format"]["duration"]),
                size_MB=round(path.stat().st_size / 1e6, 2))


res, ok = {}, True
for theme in tier["themes"]:
    fdir = ROOT / "frames" / f"{info['version']}_{info['config_hash']}_final_{theme}"  # frame folders carry the version
    if not fdir.exists():
        fdir = ROOT / "frames" / f"{info['config_hash']}_final_{theme}"  # older renders
    n_png = len(list(fdir.glob("f_*.png")))
    for ext in ("mp4", "gif"):
        p = FINAL / f"psa_n2_{theme}.{ext}"
        if not p.exists():
            res[p.name] = "missing"
            ok = False
            continue
        d = probe(p)
        if ext == "mp4":
            checks = dict(size=(d["width"], d["height"]) == tuple(tier["size"]), frames=d["frames"] == n_frames,
                          duration=abs(d["duration_s"] - video_s) < 0.1, h264=d["codec"] == "h264",
                          yuv420p=d["pix_fmt"] == "yuv420p", all_png_frames=n_png == n_frames)
        else:
            checks = dict(aspect=abs(d["width"] / d["height"] - 16 / 9) < 0.01, duration=abs(d["duration_s"] - video_s) < 0.2,
                          under_limit=d["size_MB"] <= tier["gif_max_mb"])
            d["downscaled"] = d["width"] != tier["size"][0]
            d["note"] = info["themes"][theme].get("gif_note")
        d["checks"] = checks
        ok &= all(checks.values())
        res[p.name] = d
res["labels"] = "legend rows name each species and its role (psan2viz.style.SPECIES_ROLE): N2/Ar product, O2/CO2/H2O extract"
res["pass"] = bool(ok)
(ROOT / "docs" / "gate4_qc.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
