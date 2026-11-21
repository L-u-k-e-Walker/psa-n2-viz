# psa-n2-viz: animated two-bed PSA nitrogen generator

A rendered visual of a two-bed carbon-molecular-sieve (CMS) PSA unit making N₂ from compressed
air (N₂, O₂, Ar, CO₂, H₂O): two 3D packed beds, a piping and valve schematic with flows, and live
plots, at cyclic steady state, as a seamless loop. **Work in progress**: see docs/MISSION.md (plan,
decisions, gates), docs/LOG.md (evidence), docs/REVIEWS.md (review history).

**Nothing here is measured.** Every input is a literature value (cited), a fit to literature data,
a calculation from those, or an assumption with its reason (docs/INPUTS.md). The motion is computed,
not scripted: every colour, profile, count and curve comes from the solver in `psan2viz/`.

## What is simulated

| Piece | Model | Code |
|---|---|---|
| Beds | Isothermal (20 °C), no axial pressure drop, multi-site extended Langmuir, LDF with a Darken factor on the CMS, layered bed (activated alumina at the feed end, CMS above). psa-viz's column equations generalized to five components and layers; first-order upwind on 180 CMS cells | `psan2viz/bed.py` |
| Plant | Both beds, the product receiver and linear valves (flow = Cv ΔP, bidirectional) integrated as one ODE system; the literature cycle (dual-end equalization, feed pressurization with product back-flow, production, blowdown with simultaneous product purge) | `psan2viz/plant.py` |
| Cyclic steady state | Swapped half-cycle map with vector-Aitken extrapolation of the slow groups for the bulk (sizing); for the CO₂/H₂O profiles, which take ~10⁴ cycles (the alumina layer slowly fills or dries), a frozen-bulk march: the trace subsystem integrated with the plant's own equations on one recorded full cycle, ~100x cheaper than a full cycle, alternated with full-model cycles. Converged when one full-model cycle changes the state by < 1e-6 (RMS, normalized) | `psan2viz/plant.py`, `psan2viz/trace.py` |

Main sources: Marcinek's dissertation on a CarboTech twin-bed pilot with Shirasagi MSC CT-350
(isotherms, kinetics, cycle) and NIST ISODB isotherm records (CO₂ on CT-350, water and CO₂ on
activated alumina).

Not simulated, drawn as context: the compressor, air receiver and dryer. The valve arrangement is a drawing of the model's valves (feed, vent, product and purge per bed, one equalization valve per end): it is drawn as two 3-way valves per end plus a feed-inlet valve, a product-outlet valve and the purge needle valve; the model's equalization valves appear as the path through both 3-way valves with the inlet (bottom) or outlet (top) valve shut. Every flow drawn is the model's valve flow. Not to scale: the 3D bed
geometry and particle size, molecule size and speed, and the trace-species molecule counts.

## Pipeline

```bash
# env viz (numpy, scipy, matplotlib, OVITO 3.17, ffmpeg); see environment.yml
cd model/fits && python -I fit_inputs.py          # sorbent inputs from the literature data
cd ../regression && python -I g1c_regression.py   # gate G1c (psa-viz regression)
cd ../sizing && python -I size_bed.py --out runs_n60c ...   # sizing at 60 cells (docs/LOG.md for the arguments)
cd ../css && python -I trace_css.py ../sizing/runs_n60c/iter02 runs/trace60   # converged trace profiles, 60 cells
cd .. && OPENBLAS_NUM_THREADS=8 python -I phase1_chain.py   # 180-cell sizing rounds, then the 360-cell grid run
OPENBLAS_NUM_THREADS=6 python -I phase1_final.py            # final 180-cell CSS, gates, Gate 3, draft
cd ../render && python -I packing.py              # 3D bed packing (seeded)
python -I render.py --mock                        # layout contact sheet (< 1 min)
python -I render.py --draft                       # 960x540 draft loop + contact sheet
python -I render.py --final vNN                   # only after "final vNN" in docs/REVIEWS.md
```

All look settings are in `render/config.toml`; each render saves the config it used next to its
output. Physics, frame states and molecule history are cached in `render/cache/`, so look changes
never re-run the model. Frames are resumable (`frames/`, gitignored).
