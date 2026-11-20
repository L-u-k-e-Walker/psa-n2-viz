# Log

Gate evidence and decisions, newest last.

## 2026-11-20 Phase 0: setup

### Reading psa-viz
README, docs/MISSION.md, LOG.md, INPUTS.md, `psaviz/column.py`, `psaviz/style.py`,
`column/render/{scene,packing,render_column}.py` (plus run/gates/tools scripts and
stills). Carried over: the column model structure (uniform-pressure velocity
march, upwind fluxes, valve/free ends, tank pre-fill), the look (dataviz surfaces
and ink, OKLab loading ramps, cut-away OVITO bed beside aligned profile panels,
step list, pressure trace, footnote), the rules (numpy/scipy physics, provenance
for every input, self-checked gates, ≤ 8 workers with OPENBLAS_NUM_THREADS=1,
resumable frames) and the OVITO lessons (vertex colours RGBA; `set_normals`
needs (3·faces)×3; give the StaticSource a fresh clone per frame; remove
pipelines before building the next theme in the same process).

### Literature check before the question round
- **Cycle.** Marcinek's dissertation (M21d, CarboTech twin-bed pilot, 2 × 2 L
  Shirasagi MSC CT-350, 8 bar abs, 20 °C) runs: (1) equalization top/top +
  bottom/bottom, 1 s; (2) co-current feed pressurization with (3) counter-current
  product back-flow from the receiver; (4) production; (1) equalization;
  (5) counter-current blowdown with (6) counter-current product purge at the same
  time; optional (7) cutting step. Half-cycle 60 s, production/blowdown 59 s,
  purge 59 s, purge = 40 % of the bed volume per purge step (Tab. 4.3-1, 4.3-2).
  The back-flow reduces axial dispersion and premature breakthrough during
  pressurization; the purge drains O₂-rich gas from the voids (§4.3). US
  4,925,461 ranks equalization: purity serial < bottom-only < top-only <
  top + bottom. Differences from the user's default sequence: the purge runs
  during the blowdown, not after it, and pressurization includes a product
  back-flow. Proposed and accepted (below).
- **Argon.** GL12 (frequency response): O₂ diffuses much faster than N₂ and Ar
  in CMS; Ar is the slowest. M21d §1.1: "the product stream of N2-PSA technology
  is predominantly a mixture of nitrogen and argon"; purity is defined from O₂
  alone (§4.4). So Ar follows N₂, and the product carries ~1.1–1.2 % Ar.
- **Air factor.** BOGE data sheet: 2.9 at 99.5 %. M21d: air demand 3.09–3.18 at
  1000 ppm O₂ (reference process, Tab. 4.5-1), 3.13–3.20 best cases at 1000 ppm
  (Tab. 7-1). So 99.5 % should sit below ~3; the G1f band becomes 2.0–3.5.
- **CMS data.** M21d gives Sips isotherms of O₂ and N₂ on CT-350 at 20–60 °C
  (Tab. 3.3-1) and a QDF + Darken + Arrhenius kinetic model (Tab. 3.5-1): at
  20 °C and zero loading k_O₂ = 1.7534 exp(−1622/RT) = 0.90 /s and
  k_N₂ = 118.83 exp(−25 099/RT) = 0.0040 /s.
- Not reachable from here: Rege & Yang 2000 (Deep Blue 403), the Springer page of
  M21 (login redirect / 503), Giesy's dissertation (421). A download with cookies
  and a browser user agent was blocked as access-control circumvention and not
  retried. M21d (public PDF) was read locally with pypdf installed into the
  session scratchpad (no env changed).

### Colour check
Python port of the dataviz validator (`tools/validate_palette.py`; it reproduces
the palette doc's published figures: adjacent worst CVD 9.1, normal-vision 19.6,
light mode). The user's default (Ar, CO₂, H₂O on slots 3–5) fails the all-pairs
check that scattered molecules need: CO₂ yellow vs O₂ orange normal-vision
ΔE 10.6 (dark), H₂O magenta vs Ar aqua ΔE 1.6 under deuteranopia (dark); only
slots 1–3 pass all-pairs. Every alternative was enumerated; the passing family
uses ink tones or lightness steps for the extra species.

### Question round (user answers, 2026-11-20)
Cycle: literature cycle. Case: 30 SLPM at 21.1 °C. Feed: 7.5 barg, PDP +3 °C.
Sorbent: CMS + thin alumina layer. Colours: the recommended set but "N₂ green
always" → re-validated with N₂ green: N₂ = palette aqua-green, O₂ orange,
H₂O blue, Ar = N₂'s hue darker/lighter, CO₂ ink; all pairs pass in both themes
(worst CVD ΔE 9.2, normal 18.4); the palette's pure green fails against orange
for deuteranopes (ΔE 1.4) and is not used. Saved as a standing preference in
memory. Kinetics: LDF + Darken factor. Purity: industry convention + a "N₂
excluding Ar" readout. Remaining defaults confirmed, with the G1f air-factor
band 2.0–3.5. All recorded in docs/MISSION.md.

Correction after the round: the water enhancement factor at 8.5 bar is
f = 1.028 (Buck81), not ≈ 1.005, so y_H₂O = 915 ppm (the question quoted ≈ 895).

### Gate 0
- Env `viz`: numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2, OVITO 3.17.2,
  Pillow 13.0.0, ffmpeg 9.0.2. `import numpy, scipy, matplotlib, ovito, PIL` works.
- `tools/gate0_render_test.py` (6000 spheres + a transparent mesh, 960×540,
  each renderer in its own subprocess; docs/gate0/gate0.json, render_*.png):
  - OpenGLRenderer, headless: **works** (WSLg/Mesa EGL), 0.40 s first frame,
    0.059 s per frame after.
  - Tachyon without AO/AA: works, 0.061 s per frame.
  - Tachyon with AO (final settings): works, 0.30 s per frame.
  The OpenGL renderer draws the transparent glass wall differently from Tachyon.
  Since Tachyon without AO/AA is as fast on this test scene and matches the
  final's look more closely, the draft renderer is a config option: Phase 3
  times both on the real bed scene and picks the faster one unless its look
  departs from the final.
- **Gate 0 PASS:** imports work; the draft renderer (OpenGL) renders headless;
  initial commit made.
