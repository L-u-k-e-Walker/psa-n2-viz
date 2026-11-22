"""Shared look: theme tokens, species colours, loading ramps.

Origin: theme tokens, OKLab helpers, ``ramp``, ``darken``, ``apply_ramp`` and
``mpl_rc`` are copied from psa-viz ``psaviz/style.py`` (git@github.com:L-u-k-e-Walker/psa-viz.git,
commit 709af6f), unchanged except where noted. Species colours are new.

Colours follow the dataviz reference palette (categorical slots, chart surfaces
and ink). Species (user decisions 2026-11-20, docs/MISSION.md):

* N2 green (user: "N2 green always") = the palette's aqua-green slot; the
  palette's pure green collapses against O2 orange for deuteranopes.
* O2 orange (psa-viz's A), H2O blue (slot 1).
* Ar = N2's hue stepped darker (white theme) / lighter (dark theme): "inert,
  goes with N2" (a composite encoding, not a new hue).
* CO2 = primary ink.

All 10 pairs pass the all-pairs CVD and normal-vision checks in both themes
(``python tools/validate_palette.py --species``): worst CVD ΔE 9.2, worst
normal-vision ΔE 18.4.
"""

from __future__ import annotations

import numpy as np

THEMES = {
    "white": dict(
        bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
        bare="#d8d4cb", metal="#9a9890", glass="#b9c6d2", highlight="#f0efec",
        bare_alumina="#eeebe4", tube_edge="#a9a79f", tube_fill="#ecebe6", tube_shine="#ffffff",
        valve_dark="#4f4d49", valve_mid="#b9b6ad", card="#f4f3ef",
        N2="#1baf7a", O2="#eb6834", Ar="#006644", H2O="#2a78d6", CO2="#0b0b0b",
    ),
    "dark": dict(
        bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
        bare="#4a4945", metal="#6d6b66", glass="#5d6b78", highlight="#2c2c2a",
        bare_alumina="#6b6964", tube_edge="#5a5853", tube_fill="#262624", tube_shine="#77756f",
        valve_dark="#cfcdc6", valve_mid="#77756f", card="#222220",
        N2="#199e70", O2="#d95926", Ar="#6adeac", H2O="#3987e5", CO2="#f0efec",
    ),
}

SPECIES = ("N2", "O2", "Ar", "CO2", "H2O")  # solver component order everywhere in psan2viz
SPECIES_TEX = {"N2": "N$_2$", "O2": "O$_2$", "Ar": "Ar", "CO2": "CO$_2$", "H2O": "H$_2$O"}
SPECIES_ROLE = {  # legend text: species and its role
    "N2": "product",
    "Ar": "product (follows N$_2$)",
    "O2": "extract (held by the CMS)",
    "CO2": "extract (held at the feed end)",
    "H2O": "extract (held by the alumina)",
}


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def rgb_hex(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in c)


# --- OKLab (Björn Ottosson) for perceptually even ramps
def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gam(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def rgb_to_oklab(rgb):
    lms = _lin(rgb) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_rgb(lab):
    lms = (np.asarray(lab) @ np.linalg.inv(_M2).T) ** 3
    return np.clip(_gam(lms @ np.linalg.inv(_M1).T), 0, 1)


def ramp(c0, c1, n=256):
    """n colours from c0 to c1, interpolated in OKLab. Returns (n, 3) RGB."""
    a, b = rgb_to_oklab(hex_rgb(c0)), rgb_to_oklab(hex_rgb(c1))
    t = np.linspace(0, 1, n)[:, None]
    return oklab_to_rgb(a + (b - a) * t)


def darken(h, f=0.18):
    lab = rgb_to_oklab(hex_rgb(h))
    lab[0] *= 1 - f
    return rgb_hex(oklab_to_rgb(lab))


def loading_ramp(theme, species, n=256, deep=False, bare_key="bare", strength=None):
    """Sequential one-hue ramp for a species' loading: bare sorbent -> species colour.

    ``deep`` ends on a darker step of the hue, so that molecules drawn in the
    species colour stay visible in front of fully loaded particles.
    (psa-viz original; ``bare_key`` added so the alumina layer can start from its own bare colour.)"""
    t = THEMES[theme]
    if strength is not None:  # a pale ramp: bare -> a tint ``strength`` of the way to the species colour
        end = rgb_hex(ramp(t[bare_key], t[species], 256)[int(round(float(strength) * 255))])
        return ramp(t[bare_key], end, n)
    f = (0.30 if theme == "white" else 0.22) if deep else (0.08 if theme == "white" else 0.0)
    return ramp(t[bare_key], darken(t[species], f), n)


def apply_ramp(r, x):
    """Map x in [0, 1] (array) through ramp r (n, 3)."""
    x = np.clip(np.asarray(x, float), 0, 1)
    idx = x * (len(r) - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(r) - 1)
    w = (idx - i0)[..., None]
    return r[i0] * (1 - w) + r[i1] * w


def mpl_rc(theme, scale=1.0, family="DejaVu Sans"):
    """matplotlib rc for a theme (psa-viz original; ``scale`` added for the 960x540 mock/draft tiers, ``family``
    for the typeface: maths text (subscripts) uses the same face)."""
    t = THEMES[theme]
    s = scale
    return {
        "font.family": family, "font.size": 15 * s,
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold", "mathtext.sf": family,
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.facecolor": t["bg"],
        "figure.facecolor": t["bg"], "savefig.facecolor": t["bg"], "axes.grid": True,
        "grid.color": t["grid"], "grid.linewidth": 0.8 * s, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 17 * s, "axes.titleweight": "bold", "axes.titlecolor": t["ink"],
        "axes.labelsize": 15 * s, "xtick.labelsize": 13 * s, "ytick.labelsize": 13 * s, "legend.frameon": False,
    }
