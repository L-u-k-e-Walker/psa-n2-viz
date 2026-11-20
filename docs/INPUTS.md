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

## To be filled in Phase 1

CMS isotherms (Langmuir fit to the Sips data of M21d), CMS LDF k_i0 with the
Darken factor (fit to M21d's kinetic model), argon and CO₂ on CMS, H₂O and CO₂
(and O₂/N₂/Ar) on activated alumina, bed geometry and layer height, bed void and
densities, valve constants, step durations, product-receiver volume, purge
valve, grid.
