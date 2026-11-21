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

## 2026-11-20 Phase 1: two-bed model (in progress)

### Code
- `psan2viz/isotherm.py` (multi-site extended Langmuir, LDF + Darken), `bed.py` (layered grid +
  the shared rate kernel: psa-viz's equations generalized to n components, per-cell sorbents,
  several valves per bed end), `column.py` (psa-viz's 4-step single column on the shared kernel,
  for G1c), `plant.py` (two beds + receiver + linear bidirectional valves as one ODE; the
  literature cycle; CSS), `cases.py` (the N2-generator case with provenance), `css.py`,
  `analysis.py` (cycle sampling, H2O front, equalization residual, symmetry).
- Frozen copy of psa-viz's column model in tests/psaviz_ref (origin header); nothing imports
  ../psa-viz. The one ISODB record that came from psa-sim's data folder is copied into
  model/data/isodb/.

### Inputs (docs/INPUTS.md; model/fits)
- Marcinek's Sips parameters reproduce the ISODB CT-350 N2 data at 293 K (1.068/1.460/1.834 vs
  1.065/1.45/1.835 mmol/g at 5/10/20 bar), so T0 = 293.15 K and his parameters apply at 20 C.
- Single-site Langmuir fits missed the Sips curves by up to 20 % at 6-10 bar; dual-site fits are
  within 4.4 % (O2) and 6.0 % (N2). Alumina CO2 needs two sites too (joint 273/303 K fit, 1.5 %).
- The CMS strong site (0.02 mol/kg, b ~20 /bar) would crowd out any species placed on it, so CO2
  and H2O sit on the main site.
- ISODB "CMS water 293 K, Isotherm4" exceeds the saturation pressure of water at 293 K and was not
  used.
- Water on alumina: type II; the dual-site fit is a strong site plus a linear term (6 % up to
  RH 0.5). The strong-site constant is capped at b_RH = 100 (the data start at RH 0.043 and
  cannot tell larger values apart); this cut the plant cycle time from ~45 s to ~30 s.

### G1c regression (model/regression/g1c.json): PASS
psa-viz's case run by the frozen reference and by ColumnN with five components (N2 = B, O2 = A,
Ar/CO2/H2O zero), 62 cycles each. At rtol 1e-9: purity 9.7e-10, recovery 2.6e-9, y_A profiles
3.4e-8, loadings 7.0e-8, P(t) 1.6e-8 (rel), receiver P 5.4e-10 (rel); max 7.0e-8 < 1e-6. At the
case tolerance rtol 1e-6 the difference is 1.9e-5 (integrator step-selection noise; the state
vectors differ in length, so the error norms pick different steps). Trace species stay at 1e-20.

### Performance findings
- One cycle of the 1460-state plant took ~45 s at first: per half-cycle ~9 s in finite-difference
  Jacobians, ~8 s in 474 LU factorizations, ~7 s in the RHS. The Jacobian is dense within a bed
  (the uniform-pressure velocity march couples every cell), so sparsity cannot help.
- Culprit for the frequent Jacobian refreshes: the trace species, mostly water at its steep
  alumina isotherm (production segment: 1.6 s, 6 Jacobians without CO2/H2O vs 10.7 s, 108 with).
  Smoothing the upwind and valve-direction switches did not help and was removed. A looser
  absolute tolerance on CO2/H2O (1e-6 of their feed level) helps a little and is used.
- Two parallel runs that each used 20 OpenBLAS threads oversubscribed the CPU (load 37); every
  run now pins OPENBLAS_NUM_THREADS.
- Anderson acceleration of the half-cycle map stalled at RMS ~5e-4 (the slow modes are the CO2
  loading of the CMS and the water front in the alumina, hundreds of half-cycles). Replaced by
  Jacobian-free Newton-Krylov on the swapped half-cycle map (GMRES, finite-difference
  Jacobian-vector products, backtracking line search).
- Newton-Krylov stalled at RMS ~8e-4 (GMRES never converged within 80 products; every Newton step
  failed its line search). Diagnosis from the Aitken solver's first runs: consecutive half-cycle
  changes of the CO2/H2O loadings point in opposite directions (cos ~ -0.4), because the swapped
  half-cycle map flips the sign of the bed-to-bed difference. Measured over full cycles (two
  half-maps) the changes are monotone: decay factors ~0.98-0.99 per cycle for water, ~0.8-0.9 for
  CO2. The CSS solver used from here on (`Plant.run_to_css_proj`): plain half-cycle maps, and every
  fifth map a vector-Aitken jump of the CO2 and H2O loadings by the rest of their geometric series,
  lam/(1-lam) d2 (cosine >= 0.95, lam in (0, 1), at most 3000 cycles). The gates verify the result
  with plain full cycles, so the acceleration cannot bias it.

### Model diagnostics before sizing (starting geometry: 2 x 5.2 L beds, 0.60 m CMS + 0.06 m alumina)
- Near-CSS performance at 60 CMS cells: product O2 35,700 ppm (96.4 % N2 + Ar), air factor 2.37,
  N2 recovery 51.6 %, productivity 177 Nm3/h per m3 CMS. The O2 working capacity of one bed per
  half-cycle (~0.72 mol) barely exceeds the O2 fed (~0.62 mol); Marcinek's pilot runs at a ratio
  ~1.5 for 1000 ppm. The beds must be longer for 0.5 % O2.
- Rate law check: Marcinek's QDF + Darken law with his zero-loading constants, in a copy of the
  kernel, gives 38,300 ppm (vs 35,700 with the fitted LDF + Darken): the fitted LDF is faithful,
  not the cause of the low purity.
- Grid: first-order upwind at 60 vs 120 CMS cells: product O2 35,665 vs 32,383 ppm after 12 cycles
  (-9.2 % and still falling); N2 recovery +0.08 pp. G1e (5 %) fails at 60 cells. Cause: the
  scheme's numerical dispersion (Peclet 2N = 120) is ~7x a packed-bed correlation's
  (D_L = 0.7 D_m + 0.5 d_p u_i: Pe ~900).
- van Albada TVD face values (smooth limiter) in a copy of the kernel: mass balance exact, but
  ~12x slower per cycle (85 s vs 7 s at 60 cells), and a per-species nonlinear limiter does not
  keep the face mole fractions summing to one (it would have to act on mole fractions). The
  option stays in the kernel (default upwind; tests/test_scheme.py) but is not used.
- Asked the user (G1e failing needs a decision): finer upwind grid with the strict 5 % limit
  (180 vs 360 CMS cells) chosen.
- Exact Jacobian sparsity pattern for BDF (sparse LU): same results (3.5e-7 scaled), but slower
  than dense LAPACK at 60 cells (4.3 vs 2.6 s per half-cycle) and at 180 cells (150 vs 54 s).
  Removed; dense LU kept.
- CO2 rate on the CMS: the assumed 0.01 /s let CO2 slip through kinetically (77 ppm in the
  product); changed to 0.1 x O2 = 0.15 /s (CO2 is a small linear molecule, reported on CMS as far
  faster than N2; still an assumption, docs/INPUTS.md).

### Alumina layer kinetics (first sizing iteration)
- At the starting geometry (6 cm alumina, 2.4 mm beads, water LDF 6.6e-5 /s from Knudsen diffusion in
  5.7 nm mesopores) the CSS water profile filled the alumina at 4.1-4.2 mol/kg (92 % of its feed
  loading, unchanged over the cycle) and spread through the CMS: y_H2O >= 1 % of the feed up to
  z/L 0.88, CMS water loading up to 74 % of its feed equilibrium. With a solid-side rate that slow the
  layer exchanges < 1 % of its driving force per half-cycle, so it cannot regenerate in blowdown and
  purge and a thicker layer would not help.
- A search for literature LDF values of water on activated alumina found none that could be verified
  (a search summary attributed "125-400 1/s" to US 6,605,136, but the patent text has no such value;
  not used). The patent's macropore formula, K = 60 eps_p (1 - eps_b) D_p / d_p^2, gives ~0.9 1/s
  (gas side) for the 2.4 mm beads, consistent with the Glueckauf estimate.
- Changed two assumptions, as a design choice for a PSA pretreatment layer: 14x28 mesh beads (1.2 mm)
  and macropore (molecular) diffusion control for the bidisperse alumina. Water LDF 1.52e-3 /s (23x),
  CO2 0.054, N2/O2/Ar 2.6-6.0 /s. docs/INPUTS.md states both and why.

### Sizing at 60 cells (model/sizing, runs_n60 and runs_n60b)
- Iteration 1 (0.60 m CMS, 6 cm alumina, new alumina kinetics): product O2 32,291 ppm, air factor 2.28,
  N2 recovery 53.7 %; the alumina holds the water (4.1 -> 2.1 mol/kg bottom to top), the CMS <= 6 % of its
  feed-equilibrium water, product H2O 2.8 ppm, CO2 1.5 ppm.
- Water-front criterion corrected: a fixed gas mole-fraction contour (1 % of the feed) reached the bed top
  during blowdown only because the same tiny water partial pressure reads ~8x larger at 1 bar. The front is
  now the highest point where the water loading exceeds 10 % of the local sorbent's feed-equilibrium
  loading (it must sit in the alumina), plus the CMS water < 10 % of its feed equilibrium; the gas fraction
  at the alumina/CMS interface during production is reported (docs/MISSION.md G1f).
- Iteration 2 (0.93 m CMS, 7.8 cm alumina): O2 10,150 ppm, air factor 2.78, recovery 45 %, productivity
  115 Nm3/h per m3 CMS, product CO2 0.23 ppm, H2O 2.0 ppm; loading front at the alumina top, CMS water
  <= 4.6 %. Secant slope d ln(O2)/dh = -3.52 /m -> ~1.16 m CMS for 4500 ppm at 60 cells (100 mm x ~1.25 m
  columns, the shape of modular extruded-aluminium N2 generators). Sizing continues at 1.16 m / 10 cm.
- CSS-solver evolution (all verified by plain cycles in the gates): extending the Aitken jumps to the
  receiver and bulk loadings set up a period-two oscillation (bulk removed); measuring only after two
  relaxation maps, adaptive damping and a 10-cycle jump cap made it smooth. At 7.8 cm alumina the
  remaining residual is the CO2 and water profiles drifting at a constant rate (decay 0.997-1.000 per
  cycle; bulk species converged to 1e-7-1e-6): a slow translation of the water front with CO2 displaced
  along it. Anderson mixing of the full-cycle map on the slow groups did not speed it up. Sizing uses a
  3e-5 target (O2 converged); the final state gets a Newton-Krylov solve on the trace subspace
  (`Plant.run_to_css_trace_nk`, full-cycle maps, bulk held at its CSS).
- runs_n60b iterations 1-5 (1.16 -> 1.15 -> 0.75 -> 1.06 -> 1.22 m CMS; alumina 0.10 -> 0.29 m): product O2
  4383, 3014, 16890, 7133, 4029 ppm; air factor 3.15, 3.30, 2.93, 3.43, 3.87. Two faults in the sizing loop,
  not in the model: (1) the loading criterion grew the alumina every iteration: the alumina's strong water
  site fills at RH ~1 %, so any trace of water reads as "loaded" and the front always sat at the layer top
  even when the CMS held < 1 % of its feed-equilibrium water; (2) the secant mixed a purge change
  (iteration 1 had half the target purge) with the CMS change and overshot to 0.75 m. The run was stopped
  (and the waiting 180-cell chain with it).
- Fixes: the water criterion is now "the alumina captures >= 95 % of the water" (CMS water < 5 % of feed
  equilibrium and interface y_H2O during production < 5 % of the feed; MISSION G1f updated); the secant uses
  only iterations with the purge within 5 % of target and the same alumina. Iteration 2 of runs_n60b
  (1.15 m CMS, 13.1 cm alumina, purge on target) meets the new criterion (CMS 2.3 %, interface 3.0 %);
  iteration 1 (10.1 cm) narrowly fails it (interface 5.5 %), so the alumina stays at 13.1 cm. Restarted as
  runs_n60c from 1.04 m (secant from iteration 2 with -3.52 /m), then the chain (180 cells -> 360 cells).
- runs_n60c (CSS tolerance 5e-5 for sizing: below that only the slow water mode is left, product O2 is
  settled): iteration 1, 1.039 m CMS: O2 4,927 ppm, air factor 3.15, recovery 40.0 %, purge 0.165 vs target
  0.164 mol; iteration 2, **1.064 m CMS, 13.1 cm alumina, cv_purge 4.45e-4: O2 4,503 ppm, air factor 3.18,
  N2 recovery 39.6 %, purge 0.164 vs 0.168 mol (-2.3 %), CMS water 2.0 % of feed equilibrium, interface
  y_H2O 2.6 % of the feed: sized** (60 cells). Secant slope -3.49 /m (the earlier estimate was -3.52).
  The chain took it up: upsampled to 180 + 30 cells, size_bed.py at 180 cells (tolerance 5e-5).

### Final trace convergence (G1b): single-mode deflation
- Diagnosis: near the CSS the residual is ONE slow eigenmode of the full-cycle map (water front creeping
  through the alumina with CO2 displaced along it; decay 0.9995 per cycle, ~2,000-cycle e-folding).
  run_to_css_proj extrapolates CO2, H2O and the receiver as separate groups with separate decay factors
  and a damping driven by the global residual, which splits that mode and crushed the water jumps to
  2-3 cycles; a long-baseline variant (runs/bl4, baseline 4, cap 3000) settled at rms ~2e-5 after 8 steps.
- New `Plant.run_to_css_mode` (script model/css/converge_trace.py): `relax` plain full cycles, then three
  more give consecutive whole-state changes d1, d2, d3; if aligned (cos >= 0.995) with consistent decay
  factors, the whole state jumps along d3 by the rest of the geometric series, capped so no variable
  changes by more than max_step (normalized).
- Test at 60 cells (sized iteration 1 state, rtol 1e-7, runs/mode60): cos 1.0000/1.0000, lam 0.99948/0.99954,
  jump 251 cycles (series 2,181, capped) -> residual 1.43e-5 -> 1.25e-5 as predicted (0.9995^251 = 0.88);
  steps right after a jump see the excited fast modes (lam > 1 or cos < 0.995) and skip. Working, but the cap
  (0.05) limits it to ~250 cycles per jump.
- Follow-up: with a larger cap (0.2, runs/mode60s) and over more steps (runs/mode60) the decay factors settled
  at 0.9991-1.0003, i.e. a near-constant drift, not a geometric approach; the residual stuck at 1.0e-5.
  The cycle water balance shows why: the vent removes 4.8 % more water than the feed brings (-3.5e-4
  mol/cycle; G1b's 0.5 % balance limit fails too). The alumina is loaded through its whole depth
  (3.9 -> 1.2 mol/kg over 13 cm) and is drying out: the CSS holds substantially less water, an O(1) change
  in the normalized water loadings, ~1e4 cycles away at this rate. Linear jumps can't cover it, and
  projective (forward) steps of hundreds of cycles are unstable for the intermediate modes (decay 0.9-0.97).
- Plan (grid sequencing): converge the trace profiles fully at 60 cells (cheap cycles, ~10 s) with
  Newton-Krylov on the trace subspace (`run_to_css_trace_nk`, Krylov 50, inexact eta 0.1, rtol 1e-7;
  runs/nk60), then carry them into the 180-cell sized state, where the correction is small and the
  single-mode jumps apply.
- Trace NK at 60 cells (runs/nk60) failed: GMRES did not converge in 100 iterations and the line search
  rejected even lambda = 0.1 in both Newton steps (the finite-difference Jacobian through the adaptive
  integrator and the clipping is too noisy). Stopped.
- Frozen-bulk trace march (`psan2viz/trace.py`): the CO2/H2O subsystem integrated with the bulk taken
  from one recorded full cycle of the complete model (its dense output), using the plant's own RHS (all
  valve/boundary couplings unchanged), BDF with a numerically detected sparsity pattern (entries < 1e-4
  of the row maximum dropped: the weak coupling through the velocity field). Check at 60 cells: from the
  same start, one frozen-bulk cycle reproduces the full model's trace end state to rms 2.7e-7 (the
  cycle's own change is 2.6e-5, so 1 %), the same for pattern thresholds 1e-6..1e-2; 1.06 s per cycle vs
  15 s for a full cycle (2,550 RHS calls at 0.19 ms). `trace.converge` alternates 3 full-model cycles
  (bulk follows the trace state) with marches; convergence is judged only on full-model cycles.
  Driver model/css/trace_css.py; running at 60 cells (runs/trace60).

### Sizing at 180 cells (model/sizing/runs_n180)
- Iteration 1 (the 60-cell design, 1.064 m CMS, 13.1 cm alumina, upsampled state): product O2 1,284 ppm vs
  4,503 ppm at 60 cells (air factor 3.18, recovery 39.7 %). The 60-cell upwind grid smears the O2 front
  (numerical Peclet 2N), so it under-predicts purity by 3.5x at this design: the reason for the finer
  grid (G1e). Next: 0.705 m. The chain now runs sizing rounds (runs_n180, runs_n180a, ...) until the
  acceptance holds, restarting each round from the last point with the slope measured at 180 cells.
- Trace march at 60 cells finished (runs/trace60, sized 60-cell design): 4 outer rounds, ~10,200 trace
  cycles in 2.5 h; a verified full-model cycle then changes the normalized state by 3.0e-7 rms (max 2.2e-6),
  cycle balances <= 2.1e-4 (water) -> G1b criteria met at 60 cells. Product O2 4,484 ppm (4,503 before the
  trace profiles settled: the CMS holds slightly less water at the CSS).
- 180-cell sizing round runs_n180: iteration 2 (0.705 m) stalled 6.3 h (421 half-maps, rms ~7e-5, target
  5e-5) in the same slow water mode. Stopped; the CSS solves in sizing and in the 2N run are now capped
  (40 / 30 half-maps: product O2 and recovery have settled long before) and start from the converged trace
  profiles of runs/trace60 mapped onto the target grid and CMS height (`psan2viz.trace.transfer`).
  Continued as runs_n180a from the stalled run's checkpoint.
- Frozen-bulk trace cycle at 180 cells: 2.2 s vs 250 s for a full cycle (sparsity 0.2 %), matching the full
  model to rms 4.3e-7 (cycle change 7.8e-5). The final state is converged directly at 180 cells.
- model/phase1_final.py (unattended): sized 180-cell state -> trace_css.py -> model/css/final (G1b target
  1e-6, rtol 1e-7) -> gates_phase1.py once model/grid/2N is done -> gate3_spotcheck.py -> draft on the final
  physics. Runs beside phase1_chain.py.
- Gate dry run on the converged 60-cell state (gates_phase1.py css/runs/trace60, no 2N; gates_dryrun.json):
  G1a PASS (start-up cycles 2.9e-11), G1b PASS (3.0e-7 rms, 2.2e-6 max; balances < 0.5 %), G1c PASS,
  G1d PASS (2.2e-6), G1f PASS (O2 4,484 ppm, air factor 3.18, CMS water 0.65 % of feed equilibrium,
  interface y_H2O 0.87 % of feed, product CO2 0.017 ppm, receiver >= 7.48 bar vs 7.01 delivery), G1e n/a.
  The pipeline is ready for the 180-cell state.
- Round runs_n180a, iteration 1 (0.705 m, from the stalled run's checkpoint, trace profiles from trace60):
  O2 13,177 ppm, air factor 2.74, recovery 45.7 %, but purge 0.171 vs target 0.117 mol (+46 %): the purge
  valve update kept the purge per step while the target shrank with the bed. CMS water 0.4 %. Iteration 2
  went back to 1.013 m with the 60-cell default slope (round a could not see round 0's point). Stopped.
- Sizing update rule moved to model/sizing/sizing_rule.py, shared by size_bed.py and the chain: purge valve
  rescaled by target/purge x V_new/V_old (the purge per step follows the valve and pressures, the target
  the packed volume); secant over all points of the grid (`--prior` passes earlier rounds), stepping from
  the purge-on-target point nearest the target. 180-cell slope from rounds 0 and a: -6.47 /m (60 cells:
  -3.49; sharper fronts). Round runs_n180b: 0.8705 m, cv_purge 3.73e-4.
