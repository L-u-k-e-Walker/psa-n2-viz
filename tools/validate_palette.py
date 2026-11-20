"""Categorical palette validator (Python port).

Origin: a port of ``scripts/validate_palette.js`` from the dataviz skill bundled
with Claude Code (2026), checks 2, 3, 4, 4b and 5 with the same thresholds and
the same Machado-Oliveira-Fernandes (2009) severity-1.0 CVD matrices. Ported
because node is not installed here. ΔE = Euclidean distance in OKLab x 100.

  2  lightness band   OKLCH L inside the mode's band (light 0.43-0.77, dark 0.48-0.67)
  3  chroma floor     OKLCH C >= 0.10
  4  CVD separation   min over protan/deutan of ΔE: >= 8 pass, 6-8 warn (needs secondary encoding), < 6 fail
  4b normal vision    worst unsimulated ΔE >= 15 (hard gate)
  5  contrast         WCAG ratio vs the chart surface >= 3 (below: relief rule, visible labels)

Usage:
  python tools/validate_palette.py "#1baf7a,#eb6834" --mode light --pairs all
  python tools/validate_palette.py --species        # psan2viz.style species colours, all pairs, both themes

For the species set the band and chroma checks are reported but not enforced:
Ar is a deliberate lightness step of N2's hue and CO2 is a neutral ink (both
documented exceptions, docs/MISSION.md). Pair separation (4, 4b) is enforced.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}


def _srgb(h):
    h = h.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"bad hex {h!r}")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def _s2l(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin(h):
    return [_s2l(c) for c in _srgb(h)]


def _oklab(rgb_lin):
    r, g, b = rgb_lin
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s]


def _sim(h, kind):
    r, A = _lin(h), MACHADO[kind]
    return [min(1.0, max(0.0, sum(A[i][j] * r[j] for j in range(3)))) for i in range(3)]


def delta_e(a, b, kind=None):
    x = _oklab(_sim(a, kind) if kind else _lin(a))
    y = _oklab(_sim(b, kind) if kind else _lin(b))
    return 100 * math.dist(x, y)


def oklch(h):
    L, a, b = _oklab(_lin(h))
    return L, math.hypot(a, b)


def contrast(a, b):
    def lum(h):
        r, g, bb = _lin(h)
        return 0.2126 * r + 0.7152 * g + 0.0722 * bb
    hi, lo = sorted([lum(a), lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def validate(palette, names=None, mode="light", pairs="adjacent", surface=None, enforce_band=True):
    """Returns (ok, report lines)."""
    names = names or palette
    surface = surface or SURFACE[mode]
    lo, hi = BAND[mode]
    out, ok = [], True
    off = [(n, round(oklch(c)[0], 3)) for n, c in zip(names, palette) if not lo <= oklch(c)[0] <= hi]
    low_c = [(n, round(oklch(c)[1], 3)) for n, c in zip(names, palette) if oklch(c)[1] < CHROMA_FLOOR]
    if enforce_band and (off or low_c):
        ok = False
    out.append(f"[{'PASS' if not off else ('FAIL' if enforce_band else 'NOTE')}] lightness band {lo}-{hi}: "
               f"{off or 'all inside'}")
    out.append(f"[{'PASS' if not low_c else ('FAIL' if enforce_band else 'NOTE')}] chroma >= {CHROMA_FLOOR}: "
               f"{low_c or 'all above'}")
    n = len(palette)
    pl = list(itertools.combinations(range(n), 2)) if pairs == "all" else [(i, i + 1) for i in range(n - 1)]
    cvd = min((delta_e(palette[i], palette[j], k), k, names[i], names[j]) for k in ("protan", "deutan") for i, j in pl)
    nor = min((delta_e(palette[i], palette[j]), names[i], names[j]) for i, j in pl)
    tri = min(delta_e(palette[i], palette[j], "tritan") for i, j in pl)
    state = "PASS" if cvd[0] >= CVD_TARGET else ("WARN" if cvd[0] >= CVD_FLOOR else "FAIL")
    ok &= state != "FAIL" and nor[0] >= NORMAL_FLOOR
    out.append(f"[{state}] CVD separation ({pairs}): worst {cvd[2]}-{cvd[3]} ΔE {cvd[0]:.1f} ({cvd[1]}); tritan {tri:.1f}")
    out.append(f"[{'PASS' if nor[0] >= NORMAL_FLOOR else 'FAIL'}] normal-vision floor: worst {nor[1]}-{nor[2]} ΔE {nor[0]:.1f}")
    low = [(nm, round(contrast(c, surface), 2)) for nm, c in zip(names, palette) if contrast(c, surface) < CONTRAST_MIN]
    out.append(f"[{'PASS' if not low else 'WARN'}] contrast vs {surface} >= {CONTRAST_MIN}: "
               f"{low or 'all'}{' (relief rule: visible labels)' if low else ''}")
    return ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("palette", nargs="?")
    ap.add_argument("--mode", default="light", choices=["light", "dark"])
    ap.add_argument("--pairs", default="adjacent", choices=["adjacent", "all"])
    ap.add_argument("--surface")
    ap.add_argument("--species", action="store_true", help="check psan2viz.style species colours (all pairs, both themes)")
    a = ap.parse_args()
    if a.species:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from psan2viz.style import SPECIES, THEMES
        all_ok = True
        for theme, mode in (("white", "light"), ("dark", "dark")):
            pal = [THEMES[theme][s] for s in SPECIES]
            ok, rep = validate(pal, list(SPECIES), mode, "all", THEMES[theme]["bg"], enforce_band=False)
            print(f"{theme} theme ({', '.join(f'{s} {c}' for s, c in zip(SPECIES, pal))}):")
            print("  " + "\n  ".join(rep))
            all_ok &= ok
        print("ALL PAIRS PASS" if all_ok else "FAILED")
        sys.exit(0 if all_ok else 1)
    if not a.palette:
        ap.error("give a palette or --species")
    pal = [c for c in a.palette.split(",") if c.strip()]
    ok, rep = validate(pal, None, a.mode, a.pairs, a.surface)
    print("\n".join(rep))
    print("ALL CHECKS PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
