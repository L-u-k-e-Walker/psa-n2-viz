"""Gate G1c: the n-component kernel reproduces psa-viz's column model (env viz).

psa-viz's case (generic A/B, 120 cells, 4-step Skarstrom cycle) is run with the frozen
reference copy (tests/psaviz_ref) and with psan2viz.column.ColumnN using five components
(N2, O2, Ar, CO2, H2O): N2 plays psa-viz's B (light), O2 plays A (heavy), Ar, CO2 and H2O
are zero in the feed and the initial gas, the Darken factor is off. Both codes run the
same number of cycles (psa-viz's CSS count, 62) at the case tolerance (rtol 1e-6) and at
a tight tolerance (rtol 1e-9, which removes integrator noise from the comparison).
Compared: purity and recovery of the light product, y_A(z) and q(z) at 40 samples per
step, column and tank pressure P(t). Writes g1c.json.

Run: conda run -n viz python -I g1c_regression.py [rtol ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from psaviz_ref.cases import column_case  # noqa: E402
from psaviz_ref.column import ColumnModel  # noqa: E402
from psan2viz.bed import BedGrid, Layer  # noqa: E402
from psan2viz.column import ColumnCase, ColumnN, Step  # noqa: E402
from psan2viz.isotherm import Sorbent  # noqa: E402

N2, O2 = 0, 1
N_CYCLES = 62
SAMPLES = 40


def new_case(ref, rtol):
    n = 5
    qs = np.zeros((n, 1))
    b = np.zeros((n, 1))
    k = np.ones(n)
    qs[O2, 0], qs[N2, 0] = ref.qs[0], ref.qs[1]
    b[O2, 0], b[N2, 0] = ref.b[0], ref.b[1]
    k[O2], k[N2] = ref.k_ldf[0], ref.k_ldf[1]
    sorb = Sorbent("psaviz_generic", qs, b, k, darken=False)
    lay = Layer("bed", sorb, ref.length, ref.bed_void, (1 - ref.bed_void) * ref.pellet_density, ref.n_cells)
    grid = BedGrid([lay], ref.radius, ref.temperature)
    yf = np.zeros(n)
    yf[O2], yf[N2] = ref.y_feed
    y0 = np.zeros(n)
    y0[N2] = 1.0
    steps = [Step(s.kind, s.duration, s.cv) for s in ref.steps]
    return ColumnCase(grid=grid, y_feed=yf, p_feed=ref.p_feed, p_vent=ref.p_vent, tank_volume=ref.tank_volume,
                      tank_p_set=ref.tank_p_set, p_init=ref.p_init, y_init=y0, steps=steps, light=N2, heavy=O2,
                      rtol=rtol, atol_rel=ref.atol_rel, css_tol=ref.css_tol, css_components=(N2, O2))


def compare(rtol):
    ref_case = column_case(n_cells=120, rtol=rtol)
    t0 = time.time()
    m_ref = ColumnModel(ref_case)
    _, recs_ref, hist_ref = m_ref.run_to_css()
    wall_ref = time.time() - t0
    n_css_ref = hist_ref[-1]["cycle"]
    if n_css_ref != N_CYCLES and rtol == 1e-6:
        print(f"note: psa-viz reached CSS after {n_css_ref} cycles", flush=True)
    # re-run psa-viz for exactly N_CYCLES (identical to its CSS run when n_css_ref == N_CYCLES)
    m_ref = ColumnModel(ref_case)
    x = m_ref.initial_state()
    for _ in range(N_CYCLES):
        x, recs_ref = m_ref.run_cycle(x)
    met_ref = m_ref.cycle_metrics(recs_ref)
    pr_ref = m_ref.profiles(recs_ref, SAMPLES)
    t1 = time.time()
    m_new = ColumnN(new_case(ref_case, rtol))
    _, recs_new, hist_new = m_new.run_to_css(n_cycles=N_CYCLES)
    wall_new = time.time() - t1
    met_new = hist_new[-1]
    pr_new = m_new.profiles(recs_new, SAMPLES)
    # CSS cycle count with the identical (N2, O2) norm
    _, _, hist_new_css = (None, None, None)
    res = dict(
        rtol=rtol, cycles=N_CYCLES, psaviz_css_cycles=n_css_ref, wall_psaviz_s=round(wall_ref, 1),
        wall_new_s=round(wall_new, 1),
        purity=[met_ref["purity_B"], met_new["purity_light"]],
        recovery=[met_ref["recovery_B"], met_new["recovery_light"]],
        d_purity=abs(met_ref["purity_B"] - met_new["purity_light"]),
        d_recovery=abs(met_ref["recovery_B"] - met_new["recovery_light"]),
        d_yA_maxabs=float(np.max(np.abs(pr_ref["y_A"] - pr_new["y"][:, O2, :]))),
        d_qA_maxabs=float(np.max(np.abs(pr_ref["q"][:, 0, :] - pr_new["q"][:, O2, :]))),
        d_qB_maxabs=float(np.max(np.abs(pr_ref["q"][:, 1, :] - pr_new["q"][:, N2, :]))),
        d_P_maxrel=float(np.max(np.abs(pr_ref["P"] - pr_new["P"]) / pr_ref["P"])),
        d_Ptank_maxrel=float(np.max(np.abs(pr_ref["p_tank"] - pr_new["p_tank"]) / pr_ref["p_tank"])),
        trace_species_max_y=float(np.max(np.abs(pr_new["y"][:, 2:, :]))),
        prefill_s=[met_ref["prefill_s"], met_new["prefill_s"]],
        step_balance_rel_new=met_new["step_balance_rel"],
    )
    keys = ("d_purity", "d_recovery", "d_yA_maxabs", "d_qA_maxabs", "d_qB_maxabs", "d_P_maxrel", "d_Ptank_maxrel")
    res["max_diff"] = max(res[k] for k in keys)
    res["pass_1e-6"] = res["max_diff"] < 1e-6
    return res


if __name__ == "__main__":
    rtols = [float(a) for a in sys.argv[1:]] or [1e-6, 1e-9]
    out = {}
    for r in rtols:
        out[f"rtol_{r:g}"] = compare(r)
        print(json.dumps(out[f"rtol_{r:g}"], indent=1), flush=True)
    tight = out.get("rtol_1e-09")
    out["G1c_pass"] = bool(tight["pass_1e-6"]) if tight else bool(all(v["pass_1e-6"] for v in out.values()))
    (HERE / "g1c.json").write_text(json.dumps(out, indent=1))
    print("G1c pass:", out["G1c_pass"])
