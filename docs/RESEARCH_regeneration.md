# What do current small nitrogen generators use: PSA or vacuum (VPSA)?

Question (user, 2026-11-21, v10 review): should the visual show a vacuum pump (VPSA) and be titled
"CONVENTIONAL CMS VPSA N2 GENERATOR"? Research the latest smaller-scale N2 generators (e.g. OXYMAT)
and make a best guess.

## Findings (web research, 2026-11-21)

| Source | What it says | Regeneration |
|---|---|---|
| OXYMAT, "Nitrogen PSA Generator Technology" page (oxymat.com/nitrogen-pressure-swing-adsorption-technology/; the page renders client-side, text read from the search engine's index) | Two pressure vessels with carbon molecular sieve (CMS); feed air at 5-10 bar(g); "While pressure is building up in one vessel, the other vessel is regenerating through a decrease in pressure"; "a pressure equalisation between the adsorption vessels will take place" | depressurization (PSA) |
| OXYMAT product listing (environmental-expert.com, third party) | Two molecular sieve columns; pressure built to 5-7 bar(g); "a part of the produced gas is used for regeneration"; CMS life up to 40,000 h | depressurization + product purge |
| Atlas Copco NGP+ (atlascopco.com, product and wiki pages) | "premium Pressure Swing Adsorption (PSA)"; two towers with CMS; regeneration "by releasing the pressure", a small part of the nitrogen flows back to push the oxygen out; Variable Cycle Saver control | depressurization + product purge |
| Parker MAXIGAS / NITROSource / DB series (parker.com, manual excerpt) | PSA from clean dry compressed air; one tower under pressure, "the other tower is held at low pressure where the oxygen is released" | low pressure (atmospheric vent) |
| Ingersoll Rand IRNG (ingersollrand.com) | towers A and B in "pressurized adsorption and depressurized regeneration" | depressurization |
| CAIRE / AirSep Alpha series (caireinc.com) | oxygen desorbed "through depressurization back to the atmosphere", assisted by a nitrogen purge | atmospheric + purge |
| Van Amerongen CA Technology VPSA (vendor listing) | VPSA nitrogen with an oil-free blower and vacuum regeneration, low pressure (< 1 bar), marketed for oil-free low-pressure supply and longer CMS life | vacuum (niche) |
| Market reports (MarketsandMarkets, Fortune Business Insights) | PSA is the leading N2 generation technology (~half the market); competition is on energy use (controls, better CMS); membranes for lower purity | n/a |

## Best guess

Current small and mid-size on-site nitrogen generators, OXYMAT included, are **twin-tower CMS PSA units
fed with compressed air at about 5-10 bar(g)**. They regenerate by **depressurizing to atmosphere**,
with a **product-nitrogen purge**, and **equalize pressure between the towers**. That is exactly the cycle
modelled here (Marcinek's CarboTech pilot: equalization, pressurization with product back-flow,
production, blowdown with purge). Vacuum regeneration for nitrogen exists, but only as a niche
low-pressure, oil-free blower design. It is not the conventional small-generator technology. (VPSA is
the norm for zeolite **oxygen** plants.)

**Decision:** keep the PSA model (no vacuum pump). The title is "CONVENTIONAL CMS PSA N2 GENERATOR". A
VPSA variant would need a physics change (vent pressure ~0.3 bar abs, re-sizing, new CSS) and would
not represent the typical product.

Caveats: vendor pages are marketing material; OXYMAT's own text was read from the search index
because the page did not render for the fetch tool; no vendor datasheet gave the exact valve
sequence.
