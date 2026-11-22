# Inputs and where they come from

**Nothing in this project is measured.** Every number is a literature value
(cited), *assumed* (with the reason) or *calculated* from other inputs.
Code: `psan2viz/cases.py` (from Phase 1). This file is completed in Phase 1;
the entries below are the ones fixed by the user's decisions (docs/MISSION.md).

## Sources

| Key | Source | Used for |
|---|---|---|
| M21d | A. Marcinek, *Modelling and Simulation of Twin-Bed Pressure Swing Adsorption Plants for the Generation of High-Purity Nitrogen*, Dr.-Ing. dissertation, Universität Duisburg-Essen, 2021 (oral exam 2021-10-27). Public PDF: https://www.uni-due.de/imperia/md/content/verfahrenstechnik/dissertation_marcinek.pdf | Cycle (Tab. 4.3-1, 4.3-2), pilot plant (Tab. 4.2-1), CMS Shirasagi MSC CT-350 (Tab. 4.1-1), Sips isotherms of O₂/N₂ (Tab. 3.3-1), kinetics (Tab. 3.5-1, Eq. 3.5-6), air demand (Tab. 4.5-1, Tab. 7-1, 7-2), purity convention (§1.1, §4.4) |
| M21 | A. Marcinek, A. Möller, J. Guderian, D. Bathen, "Dynamic simulation of high-purity twin-bed N2-PSA plants", *Adsorption* 27 (2021) 1149–1173, doi:10.1007/s10450-021-00320-0 | Journal version of M21d (the publisher page could not be fetched; values are taken from M21d) |
| P461 | US Patent 4,925,461, "Process for separating nitrogen gas by pressure swing adsorption system" | Equalization ranking: purity rises serial < bottom-only < top-only < top + bottom |
| BOGE | BOGE nitrogen generators, US data sheet: https://www.boge.com/f/287325279136465/x/8cd3fb3caa/boge-us-data-sheet-nitrogen-generators.pdf | Compressed-air factor 2.9 at 99.5 % (G1f band) |
| GL12 | T. J. Giesy, M. D. LeVan et al., "Mass transfer rates of argon, oxygen and nitrogen in carbon molecular sieve measured by pressure-swing frequency response", AIChE Annual Meeting 2012, paper 548c (abstract) | O₂ much faster than N₂ and Ar in CMS; Ar the slowest (Ar goes with the N₂ product) |
| IF97 | IAPWS Industrial Formulation 1997, saturation-pressure equation (region 4) | p_sat of water at the dew point |
| Buck81 | A. L. Buck, "New equations for computing vapor pressure and enhancement factor", *J. Appl. Meteorol.* 20 (1981) 1527–1532 | Enhancement factor f of water in compressed air (pressure term extrapolated to 8.5 bar) |
| ISO8573 | ISO 8573-1:2010, compressed-air purity classes | Class 4 humidity: pressure dew point ≤ +3 °C (refrigerated dryer) |

## Feed and product (user decisions)

| Input | Value | Source |
|---|---|---|
| Product flow | 30 SLPM at 21.1 °C, 1 atm = 0.02071 mol/s (24.145 L/mol) = 1.671 Nm³/h at 0 °C | user |
| Product spec | O₂ ≤ 0.5 % (Ar counted with N₂) | user; industry convention (M21d §4.4) |
| Feed header | 7.5 barg = 8.513 bar abs, constant | user |
| Vent | 1.01325 bar (atmosphere, through a silencer) | user |
| Dry-air composition | O₂ 20.946 %, Ar 0.934 %, CO₂ 420 ppm, N₂ balance (78.078 %) | standard dry air with CO₂ at 420 ppm (user: CO₂ ~420 ppm, Ar 0.93 %) |
| Feed water | y_H₂O = f p_sat(T_dp)/P = 1.028 × 758.1 Pa / 851 325 Pa = **915 ppm**; dry fractions scaled by (1 − y_H₂O) | **calculated**: dryer pressure dew point +3 °C (ISO8573 class 4, user), p_sat from IF97, f from Buck81 |
| Temperature | 20 °C (293.15 K), isothermal | assumed: Marcinek's reference temperature (M21d Tab. 4.3-2); isothermal by user decision |

## Sorbents (model/fits/fit_inputs.py → model/fits/fits.json, figure fits.png)

Isotherms are extended (competitive) multi-site Langmuir, q*_i = Σ_s qs_is b_is p_i /
(1 + Σ_j b_js p_j), p in bar, q in mol/kg, at 293.15 K. Additional ISODB sources (NIST
adsorption database records, copied into model/data/isodb/): CT-350 CO₂
10.1007/s10450-016-9821-x; CMS O₂/N₂ 10.1002/cite.201300063; CMS water
10.1021/la061140a; alumina F-200 water and CO₂ 10.1021/la901107s and
10.1007/s10450-017-9907-0; alumina N₂ 10.1007/s10934-011-9494-5.

### CMS (Shirasagi MSC CT-350)

| Input | Value | Source |
|---|---|---|
| O₂ isotherm | site 0: qs 0.0173, b 17.84 /bar; site 1: qs 2.540, b 0.1615 /bar | **fitted** (dual-site) to M21d's Sips isotherm (Tab. 3.3-1, T₀ = 293.15 K) over 0.005–10 bar, max deviation 4.4 %; 5.3 % from the ISODB O₂ points (298 K). A single site misses Sips by up to 20 % at 6–10 bar, so the data need the second site |
| N₂ isotherm | site 0: qs 0.0266, b 21.64; site 1: qs 1.936, b 0.2351 | **fitted** as above, max deviation 6.0 %; 4.3 % from the ISODB N₂ points (293 K) |
| Ar isotherm | = O₂ | **assumed**: no ambient-temperature Ar isotherm on CMS was reachable (Ma et al. 1991 and Rege & Yang 2000 are paywalled); Ar and O₂ have nearly equal polarizability and similar isotherms on carbons |
| CO₂ isotherm | site 1: qs 3.700, b 1.570 /bar | **fitted** to ISODB CT-350 293 K (0.049–20 bar), RMS deviation 8.5 % (max 35 % at the lowest point). The feed partial pressure (3.6 mbar) lies below the data: Henry-region extrapolation |
| H₂O isotherm | site 1: qs 7.1, b 18.59 /bar | **fitted**: Henry slope of ISODB la061140a (CMS, 293 K) at RH ≤ 0.21 (max deviation 9.8 %), capacity = its highest loading. The pore-filling rise above RH ≈ 0.3 is not represented; G1f checks that water never reaches the CMS |
| Site placement | CO₂ and H₂O on site 1 | site 0 is a tiny strong site of O₂/N₂/Ar whose large b would crowd out anything placed there |
| Kinetics | LDF with Darken factor, k_i = k_i0/(1 − θ), θ = Σ q_i/qs_i; k0: O₂ 1.515, N₂ 0.00329, Ar 0.00329, CO₂ 0.1515, H₂O 0.01 /s | O₂, N₂ **fitted**: a CMS pellet under a square-wave air swing (8.0/1.0 bar, 59 s each) follows M21d's QDF + Darken + Arrhenius model (Tab. 3.5-1, Eq. 3.5-6; zero-loading QDF rates at 293 K: O₂ 0.90, N₂ 0.0040 /s). Cyclic loading swings match (O₂ exact, N₂ 2 % low); the N₂ mean loading sits up to 0.57 swing-amplitudes lower because LDF and QDF shapes differ. Ar = N₂ **assumed** (literature Ar/N₂ rate ratios on CMS lie on both sides of 1: GL12 finds Ar slowest; a DTIC report abstract, ADA223757, lists relative diffusivities O₂ 160, N₂ 8, Ar 13). CO₂ = 0.1 × O₂ = 0.15 /s **assumed** (a small linear molecule, reported on CMS as far faster than N₂ but slower than O₂; no rate constant for CT-350 was reachable; a first guess of 0.01 /s let CO₂ slip through the CMS kinetically). H₂O 0.01 /s **assumed** (water stays in the alumina; G1f checks it) |
| Bulk density, voidage | 711 kg/m³; total voidage 0.543 (inter-particle 0.404 + pellet pores) | M21d Tab. 3.1-1. The pore gas is taken in equilibrium with the bulk gas (it counts as gas capacity) |

### Activated alumina layer (F-200-like)

| Input | Value | Source |
|---|---|---|
| H₂O isotherm | site 0: qs 3.248, b 4275 /bar; site 1: qs 500, b 0.958 /bar (a linear term, 479 mol/kg/bar) | **fitted** (dual-site in RH) to ISODB la901107s (F-200, **303 K**) at RH ≤ 0.5, max deviation 6 %, then **moved to 293.15 K by assuming the loading depends on RH only** (b = b_RH/p_sat(293.15 K)). The strong-site b_RH is capped at 100: the data start at RH 0.043, where that site is already full, so larger values fit no better (4.0 % vs 6.0 %) and only stiffen the model |
| CO₂ isotherm | site 0: qs 0.443, b 40.14 /bar (39.5 kJ/mol); site 1: qs 1.844, b 0.541 /bar (27.0 kJ/mol) | **fitted** jointly to ISODB 10.1007/s10450-017-9907-0 at 273 and 303 K (max deviation 1.5 %), evaluated at 293.15 K. The feed CO₂ (3.6 mbar) lies below the data (≥ 37 mbar): Henry-region extrapolation |
| N₂, O₂, Ar isotherm | site 1: qs 1.933, b 0.2019 /bar | N₂ **fitted** to ISODB 10.1007/s10934-011-9494-5 (alumina, **313 K**, max deviation 2 %), b **moved to 293.15 K with an assumed 15 kJ/mol** heat of adsorption; O₂ and Ar **assumed** equal to N₂ |
| Site placement | water and CO₂ compete on site 0 | water displaces CO₂ from the strong sites (physically expected on wet alumina); N₂/O₂/Ar sit on site 1 |
| Kinetics | LDF: N₂ 6.0, O₂ 3.65, Ar 2.63, CO₂ 0.054, H₂O 0.00152 /s | **calculated**: Glueckauf, k = 15 D_e/(R_p² (ε_p + ρ_p Δq/Δc)), D_e = ε_p D_m/τ (activated alumina is bidisperse: molecular diffusion in the macropores controls), chord from a clean bead to the feed partial pressure |
| Bead | radius 0.6 mm (14×28 mesh), porosity 0.55, density 1250 kg/m³, tortuosity 3 | **assumed**: F-200-like 14×28 mesh beads, the small size used in fast-cycle PSA pretreatment layers. The first assumption (7×12 mesh, 2.4 mm, Knudsen diffusion in the 5.7 nm mesopores) gave a water LDF of 6.6e-5 /s; at 120 s cycles the layer could not regenerate, sat at 92 % of its feed loading and passed the water into the CMS (front at z/L 0.88), see docs/LOG.md |
| Bed packing | inter-particle void 0.40, bulk density 750 kg/m³, total voidage 0.73 | **assumed** (F-200 data sheets give ~770 kg/m³ bulk) |

## Sized and assumed plant values

The full table with values and sources is generated from the final state: docs/DESIGN.md
(tools/design_table.py on model/css/final). How each was set:

| Quantity | Value (180-cell sizing) | Source class | How |
|---|---|---|---|
| Column ID | 100 mm | assumed | typical of small CMS generators |
| CMS height | 0.905 m per bed | sized | product O2 = 4,500 ppm (10 % under the 5,000 ppm spec) at 30 SLPM; secant in ln O2 (model/sizing, sizing_rule.py); 180 cells |
| Alumina height | 13.1 cm per bed | sized | the alumina captures >= 95 % of the water (CMS water < 5 % of its feed equilibrium, interface y_H2O < 5 % of the feed during production); set at 60 cells, holds at 180 (CMS water 0.6 %) |
| Purge valve | 3.89e-4 mol/(s bar) | sized | purge of 40 % of the packed volume (normal conditions) per step, M21d Tab. 4.3-2 |
| Equalization valves | from eq_cv_for | sized | <= 2 % residual pressure difference after the 1 s step |
| Feed, product, vent valves | 0.12, 0.06, 0.15 mol/(s bar) | assumed | beds peak near 8 bar abs (M21d's reference pressure); blowdown in a few seconds |
| Receiver | 6 x one bed volume | assumed | M21d's pilot: 12 L receiver for 2 L beds |
| Delivery pressure | 6.0 barg | assumed | regulator about 1.5 bar below the inlet |
| Grid | 30 alumina + 180 CMS cells, first-order upwind | user decision | checked against 60 + 360 (G1e); 60 cells over-predicts product O2 ~3.5x at the same bed |


## Rates and energy shown in the visual (psan2viz/performance.py)

Units: litres and SLPM throughout (user, 2026-11-22); SLPM at 21.1 degC and 1 atm (24.145 L/mol), the user's
standard for the 30 SLPM product.

| Quantity | Definition | Source class |
|---|---|---|
| Product rate | delivered product flow of the CSS cycle, in SLPM | computed (the 30 SLPM is the user's spec) |
| Feed air | air drawn from the feed header per cycle, in SLPM; air factor = feed air / product | computed |
| Pressure work | ideal isothermal compression of that feed air from 1 atm to the 7.5 barg header at the model temperature (20 degC): W = n_feed R T ln(p_header / p_atm); also per litre of product (Wh per L N2) | calculated; the thermodynamic minimum, a real compressor needs more (adiabatic losses, motor and drive efficiency), not estimated |
| Productivity | product SLPM per litre of CMS in both beds | computed |

The center charts show the gas concentration of each species along the whole bed, normalized by its feed
concentration: C_i / C_i,in with C_i,in = y_i,feed p_feed / (R T). Values above 1 occur for N2 (and Ar) near
the product end, where the gas is enriched in N2 at nearly the feed pressure.

## Heat of adsorption shown in the visual (psan2viz/thermal.py)

The model is isothermal (user decision); temperature is not simulated. The thermal chart shows the heat each bed's
sorbent releases on uptake and takes up on release, Q = sum dH_i m_cell dq_i/dt from the solver's own uptake rates; in
the isothermal model it is assumed exchanged with the bed and the column wall. The adiabatic swing quoted on the chart
is the range of the cumulative heat over a cycle divided by the sorbent heat capacity (an estimate, not simulated).

| Quantity | Value | Source class |
|---|---|---|
| dH O2, N2 on CMS | 15 kJ/mol | assumed constant from M21d Fig. 3.4-1 / Tab. 3.4-1 (Clausius-Clapeyron on CT-350: <= 17 kJ/mol at minimum loading, falling with loading) |
| dH Ar on CMS | 15 kJ/mol | assumed = O2 (as the isotherm) |
| dH CO2, H2O on CMS | 25, 44 kJ/mol | assumed (typical for carbons; heat of vaporization); traces on the CMS |
| dH N2, O2, Ar on alumina | 15 kJ/mol | the model's own assumption (alumina isotherm moved to 293 K with 15 kJ/mol) |
| dH CO2 on alumina | 33 kJ/mol | mean of the fitted site heats 39.5 / 27.0 kJ/mol (ISODB 10.1007/s10450-017-9907-0, 273/303 K) |
| dH H2O on alumina | 44.2 kJ/mol | heat of vaporization at 20 degC (IF97), implied by the model's RH-only temperature shift of the water isotherm |
| Sorbent heat capacity | 880 J/(kg K) | M21d (CMS); assumed for the alumina too |
