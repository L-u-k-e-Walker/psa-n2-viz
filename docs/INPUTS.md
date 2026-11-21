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
| Kinetics | LDF with Darken factor, k_i = k_i0/(1 − θ), θ = Σ q_i/qs_i; k0: O₂ 1.515, N₂ 0.00329, Ar 0.00329, CO₂ 0.01, H₂O 0.01 /s | O₂, N₂ **fitted**: a CMS pellet under a square-wave air swing (8.0/1.0 bar, 59 s each) follows M21d's QDF + Darken + Arrhenius model (Tab. 3.5-1, Eq. 3.5-6; zero-loading QDF rates at 293 K: O₂ 0.90, N₂ 0.0040 /s). Cyclic loading swings match (O₂ exact, N₂ 2 % low); the N₂ mean loading sits up to 0.57 swing-amplitudes lower because LDF and QDF shapes differ. Ar = N₂ **assumed** (literature Ar/N₂ rate ratios on CMS lie on both sides of 1: GL12 finds Ar slowest; a DTIC report abstract, ADA223757, lists relative diffusivities O₂ 160, N₂ 8, Ar 13). CO₂, H₂O **assumed** (slow micropore uptake, minutes; neither reaches the product) |
| Bulk density, voidage | 711 kg/m³; total voidage 0.543 (inter-particle 0.404 + pellet pores) | M21d Tab. 3.1-1. The pore gas is taken in equilibrium with the bulk gas (it counts as gas capacity) |

### Activated alumina layer (F-200-like)

| Input | Value | Source |
|---|---|---|
| H₂O isotherm | site 0: qs 3.248, b 4275 /bar; site 1: qs 500, b 0.958 /bar (a linear term, 479 mol/kg/bar) | **fitted** (dual-site in RH) to ISODB la901107s (F-200, **303 K**) at RH ≤ 0.5, max deviation 6 %, then **moved to 293.15 K by assuming the loading depends on RH only** (b = b_RH/p_sat(293.15 K)). The strong-site b_RH is capped at 100: the data start at RH 0.043, where that site is already full, so larger values fit no better (4.0 % vs 6.0 %) and only stiffen the model |
| CO₂ isotherm | site 0: qs 0.443, b 40.14 /bar (39.5 kJ/mol); site 1: qs 1.844, b 0.541 /bar (27.0 kJ/mol) | **fitted** jointly to ISODB 10.1007/s10450-017-9907-0 at 273 and 303 K (max deviation 1.5 %), evaluated at 293.15 K. The feed CO₂ (3.6 mbar) lies below the data (≥ 37 mbar): Henry-region extrapolation |
| N₂, O₂, Ar isotherm | site 1: qs 1.933, b 0.2019 /bar | N₂ **fitted** to ISODB 10.1007/s10934-011-9494-5 (alumina, **313 K**, max deviation 2 %), b **moved to 293.15 K with an assumed 15 kJ/mol** heat of adsorption; O₂ and Ar **assumed** equal to N₂ |
| Site placement | water and CO₂ compete on site 0 | water displaces CO₂ from the strong sites (physically expected on wet alumina); N₂/O₂/Ar sit on site 1 |
| Kinetics | LDF: N₂ 0.253, O₂ 0.145, Ar 0.099, CO₂ 0.00225, H₂O 6.6e-5 /s | **calculated**: Glueckauf, k = 15 D_e/(R_p² (ε_p + ρ_p Δq/Δc)), D_e = ε_p/τ × Bosanquet(Knudsen in 5.7 nm pores, molecular diffusion at 4.5 bar), chord from a clean bead to the feed partial pressure |
| Bead | radius 1.2 mm, porosity 0.55, density 1250 kg/m³, pore diameter 5.7 nm, tortuosity 3 | **assumed**: F-200-like 7×12 mesh beads (pore diameter = 4 × 0.5 cm³/g / 350 m²/g) |
| Bed packing | inter-particle void 0.40, bulk density 750 kg/m³, total voidage 0.73 | **assumed** (F-200 data sheets give ~770 kg/m³ bulk) |

## To be filled after sizing

Bed geometry and layer heights, valve constants, purge valve, receiver volume, grid.
