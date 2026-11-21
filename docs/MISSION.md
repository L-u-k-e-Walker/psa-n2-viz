# psa-n2-viz mission

**Status (2026-11-20): Phase 0 in progress.**

Source of truth for this project. A rendered visual of a **two-bed PSA nitrogen
generator** making N₂ from compressed air (N₂, O₂, Ar, CO₂, H₂O): two 3D packed
beds (OVITO) with a P&ID-style piping and valve overlay and live plots, at
cyclic steady state (CSS), as a seamless loop. Built in the style of the sibling
repo `psa-viz` (its approach, look, rules and lessons), but self-contained.

The main workflow change from psa-viz: **rendering happens in tiers**. Fast,
low-resolution mocks and drafts come first and the user reviews each one; the
slow high-resolution render runs only when the user says "final vNN" for an
approved draft.

## Decisions (user, 2026-11-20)

Asked in one question round after a literature check (evidence in docs/LOG.md,
sources in docs/INPUTS.md).

| Topic | Decision |
|---|---|
| Sorbent | Carbon molecular sieve (CMS) with kinetic O₂/N₂ selectivity (O₂ fast; N₂ and Ar slow; similar isotherms), data for Shirasagi MSC CT-350 (Marcinek 2021). A thin activated-alumina layer at the feed end takes the H₂O (and some CO₂) |
| Purity convention | Industry convention: the spec is residual O₂ ≤ 0.5 % ("99.5 % N₂"), Ar counted with N₂. Readouts: product O₂ (ppm), Ar (%), CO₂ (ppm), H₂O (ppm), **plus "N₂ excluding Ar"** (true N₂). Literature check: Ar is the slowest of O₂/N₂/Ar in CMS, and the N₂-PSA product is N₂ + Ar (Marcinek 2021 §1.1, §4.4), so the product carries ~1.1–1.2 % Ar and a true N₂ ≥ 99.5 % is not reachable on CMS |
| Case size | **30 SLPM** of product at O₂ ≤ 0.5 %, standard conditions **21.1 °C, 1 atm** (24.145 L/mol: 0.02071 mol/s = 1.671 Nm³/h at 0 °C). Bed size, cycle time and pressures realistic for a small CMS generator (first estimate ~2 × 5 L of CMS). The prompt said "sized to my N₂ study"; the only N₂ study in the workspace (psa-sim `studies/n2_psa`) targets 1700 SLPM at 0 °C, so this was asked; the user chose 30 SLPM at 21.1 °C |
| Feed | Fixed header **7.5 barg** (8.513 bar abs) after a refrigerated dryer with pressure dew point **+3 °C** (ISO 8573-1 class 4): y_H₂O = f·p_sat(3 °C)/P = 1.028 × 758.1 Pa / 851 325 Pa = **915 ppm** (p_sat from IAPWS-IF97, enhancement factor f from Buck 1981; the question round quoted ≈ 895 ppm with f ≈ 1.005, corrected here). Dry air: O₂ 20.946 %, Ar 0.934 %, CO₂ 420 ppm, N₂ balance. Compressor, receiver and dryer are drawn as context, not simulated. Vent to atmosphere (1.01325 bar) through a silencer |
| Cycle | The literature cycle (Marcinek 2021, Tab. 4.3-1/4.3-2, CarboTech twin-bed pilot; US 4,925,461 ranks top+bottom equalization best for purity). Two beds half a cycle apart. Per bed: **equalization up** (top/top + bottom/bottom, ~1 s) → **feed pressurization with product back-flow** (the bidirectional product valve lets receiver gas flow back into the bed top until the bed passes the receiver pressure) → **production** → **equalization down** → **counter-current blowdown with a simultaneous counter-current product purge** from the receiver (needle valve). Half-cycle ~60 s, cycle ~120 s (final values set in Phase 1) |
| Thermal | Isothermal (as psa-viz), at Marcinek's reference temperature 20 °C; said on screen |
| Physics | psa-viz's column model generalized to n components with per-cell sorbent properties (layered bed); extended Langmuir (dual-site only where the literature data need it); **LDF with a Darken factor** on the CMS kinetics, k_i = k_i0 / (1 − θ_total), k_i0 fitted to Marcinek's kinetic model (user choice over constant LDF; the factor is switched off for the psa-viz regression G1c); plain LDF on the alumina; no pressure drop; linear valves (molar flow = Cv ΔP). Both beds and the product receiver are integrated as one ODE system. Product leaves the receiver through an outlet regulator at a constant 30 SLPM |
| Inputs | Literature values where they exist, each cited in docs/INPUTS.md; otherwise assumed (with a reason) or calculated. Nothing is measured; the README and the on-screen footnote say so |
| Species colours | **N₂ is always green** (user; standing preference). Validated set (dataviz palette, all 10 pairs, both themes, `tools/validate_palette.py`; worst CVD ΔE 9.2, worst normal-vision ΔE 18.4): N₂ green = palette aqua-green `#1baf7a` white / `#199e70` dark; O₂ orange `#eb6834` / `#d95926` (psa-viz's A); H₂O blue `#2a78d6` / `#3987e5`; Ar = N₂'s hue stepped darker `#006644` (white) / lighter `#6adeac` (dark), so it reads "inert, goes with N₂"; CO₂ primary ink `#0b0b0b` / `#f0efec`. The palette's pure green (slot 6) is not used: it collapses against O₂ orange for deuteranopes (ΔE 1.4). The user's original default (Ar, CO₂, H₂O on slots 3–5) failed the all-pairs check. Particle ramps: CMS bare → O₂ orange; alumina bare → H₂O blue |
| Trace species | Drawn at exaggerated counts so they are visible; the per-species weights are stated in the footnote. Plots show Ar, CO₂ and H₂O on a log ppm axis |
| Output | Final: 1920×1080, 30 fps, ~30 s loop (one CSS cycle), MP4 (H.264, yuv420p) + GIF, dark and white. The GIF is downscaled only if it would exceed 50 MB (and then it says so) |
| Repo | New repo `psa-n2-viz/` next to psa-viz. Remote git@github.com:L-u-k-e-Walker/psa-n2-viz.git (the user creates it on the website). **Ask before pushing** |
| Layout | Proposed as 2–3 mock options in Phase 2; the user picks one |

## Rules

- Self-contained: the physics package `psan2viz/` needs only numpy and scipy;
  rendering adds matplotlib, OVITO and ffmpeg. Env `viz` (reused). Code copied
  from psa-viz carries its origin in a header comment; nothing imports
  `../psa-viz`.
- Every input has a source in docs/INPUTS.md.
- Every colour, profile, count and curve comes from the solver; nothing is
  scripted for effect.
- The not-to-scale items are listed in the README and in the on-screen
  footnote: geometry, particle size, molecule speed and trace-species counts.
- At most 8 parallel processes (P-cores), `OPENBLAS_NUM_THREADS=1` in each worker.
- Self-checked gates: check each gate, log the evidence in docs/LOG.md, commit
  and continue. Feature branch per phase, merged into main. Stop only where a
  phase says **Stop**, or if a gate fails in a way that needs a user decision.
- Raw frames and previews stay out of git (see Render tiers).

## Render tiers

One renderer (`render/render.py`), one config file (`render/config.toml`)
holding every look setting, three quality levels.

| Tier | Command | What it makes | Target time |
|---|---|---|---|
| mock | `render.py --mock` | matplotlib only (beds as flat rectangles coloured from the solver); 960×540 stills at 6–8 key times on one contact-sheet PNG; for the layout | < 1 min |
| draft | `render.py --draft` | the full loop at 960×540, 15 fps, one theme, MP4 only, "DRAFT vNN" watermark in a corner, plus an 8-frame contact sheet. Fast 3D: OVITO OpenGLRenderer headless (EGL), else Tachyon without ambient occlusion and antialiasing | a few minutes; measured and reported |
| final | `render.py --final vNN` | the approved draft's config at 1920×1080, 30 fps, Tachyon with ambient occlusion, both themes, MP4 + GIF, no watermark | whatever it takes |

- Each render saves an exact copy of the config it used, `config_<hash8>.toml`
  (hash = first 8 hex digits of its SHA-256), next to its output.
- Mocks and drafts go to `previews/vNN/`. That folder is gitignored except the
  contact sheets and the config snapshots (a few KB, committed so an approved
  version can be re-rendered from a clean clone).
- After each mock or draft: **stop** and report the file paths, what changed
  since the last version, and the render time; wait for the verdict; log the
  version and the verdict in docs/REVIEWS.md.
- `--final vNN` refuses to run unless docs/REVIEWS.md marks vNN approved after
  the user said "final vNN". It reuses that version's config snapshot
  unchanged; only the quality tier differs.
- A draft is an exact subsample of its final: frame states and molecule history
  are computed on the final's 30 fps timeline and the draft takes every second
  frame.
- The physics is computed once and cached: the CSS state (keyed by the case),
  the states at frame times (keyed by case + timing) and the molecule history
  (keyed by those + molecule settings). Renders read the cache, so a change to
  the look never re-runs the model.
- Frames can be resumed (existing frames are skipped). Frames live under a
  folder named by the config hash, so a changed config never reuses stale frames.

## Phases and gates

### Phase 0: setup
Repo, `.gitignore` (frames/, previews/ except contact sheets and config
snapshots), env `viz`, docs (MISSION, INPUTS, LOG, REVIEWS).
**Gate 0:** the imports work; OVITO renders a test frame headless with the
draft renderer (report which renderer worked); initial commit made.

### Phase 1: two-bed model
- **G1a mass balance:** per step and per component,
  |in − out − Δinventory| < 1e-4 of that component's feed per cycle. Trace
  species are measured against their own feed.
- **G1b CSS:** the cycle-to-cycle change is < 1e-6, and the cycle balance closes
  within 0.5 % for each component.
- **G1c regression:** with Ar, CO₂ and H₂O set to zero, one bed, the Darken
  factor off, and psa-viz's case and 4-step cycle, the new model matches
  psa-viz's column.py (a frozen copy in tests/) within 1e-6 (purity, recovery,
  profiles, P(t)).
- **G1d symmetry:** at CSS, bed 2 at time t equals bed 1 at t + t_cycle/2
  (max relative difference < 1e-4).
- **G1e grid:** going from N to 2N cells changes product O₂ by < 5 % (relative)
  and N₂ recovery by < 1 pp. User decision (2026-11-20, after 60 cells failed): keep psa-viz's
  first-order upwind scheme and refine the grid: N = 180 CMS cells (+ 30 alumina cells),
  checked against 360 (+ 60); about 8-10 h of compute accepted.
- **G1f sanity:** product O₂ ≤ 0.5 %; the air factor (feed air / product, molar)
  in 2.0–3.5 (BOGE datasheet 2.9 at 99.5 %, Marcinek 3.1–3.2 at 99.9 %; flag it
  if outside); the H₂O front stays inside the alumina layer (definition: the alumina captures ≥ 95 %
  of the water: the CMS holds < 5 % of its feed-equilibrium water and the gas reaching the
  alumina/CMS interface during production carries < 5 % of the feed's water. A loading threshold
  was tried first and dropped: the alumina's strong water site fills at RH ~1 %, so any trace of
  water marks a cell as loaded) and the product CO₂ is negligible (< 1 % of
  the feed's 420 ppm; state the ppm); the bed pressures are nearly equal at the end
  of equalization (|ΔP| at the end ≤ 5 % of |ΔP| at the start; state the value).
- **psa-sim cross-check:** psa-sim (env `psa`) supports several columns,
  equalization and more than two components, but not this full case: it has no
  simultaneous top+bottom equalization step, no layered bed, no bidirectional
  product valve (back-flow) and no Darken LDF. As instructed, the full-case
  cross-check is skipped and the report says so.

Definitions used in the gates and readouts: recovery of N₂ = product N₂ / feed
N₂ (also reported for N₂ + Ar); air factor = moles of feed air / moles of
product per cycle; productivity = product Nm³/h per m³ of CMS (both beds).

### Phase 2: layout mockups
2–3 layout options as mock contact sheets with real solver data.
**Stop:** the user picks one.

### Phase 3: draft iterations
Packing for two beds, the OVITO scene, the composite.
**Gate 3:** the plotted values match the solver output (spot-check 3 frames,
max abs diff < 1e-6). Then render draft v01. **Stop** after each draft and work
the user's notes into the next one (v02, v03, …).

### Phase 4: final
Only when the user says "final vNN".
**Gate 4 (ffprobe):** 1920×1080 with the expected frame count and duration;
H.264, yuv420p, no missing frames; GIF ≤ 50 MB or downscaled (said so); the
legends name each species and its role (product or extract).

### Phase 5: handoff
README (outputs, what is simulated and what is not to scale, the pipeline, the
re-render commands, the provenance statement), update ../CLAUDE.md, final
commit. Ask before pushing.
