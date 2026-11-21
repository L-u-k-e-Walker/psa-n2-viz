"""The sizing acceptance and update rule, shared by size_bed.py and model/phase1_chain.py (numpy-free)."""
import math

O2_TARGET = 4500.0  # ppm (10 % under the 5000 ppm spec)
O2_TOL = 0.03
PURGE_TOL = 0.03


def sized(rec):
    return (abs(rec["O2_ppm"] / O2_TARGET - 1) < O2_TOL
            and abs(rec["purge_step_mol"] / rec["purge_target_mol"] - 1) < PURGE_TOL
            and rec["h2o_cms_max_rel"] < 0.05 and rec["h2o_interface_y_rel"] < 0.05)


def water_ok(rec):
    return rec["h2o_cms_max_rel"] < 0.05 and rec["h2o_interface_y_rel"] < 0.05


def next_params(history, default_slope):
    """Next (h_cms, h_al, cv_purge) after the last record of ``history`` (records of one grid).

    * h_al grows by 30 % unless the alumina captures >= 95 % of the water.
    * h_cms: secant in ln O2 over the records with the purge within 5 % of its target and the current
      alumina height (a purge or alumina change also moves O2 and would corrupt the slope); with
      fewer than two such records, ``default_slope``. Step limited to 0.5x..1.6x.
    * cv_purge: the purge per step scales with the valve (it is set by the receiver and bed
      pressures, nearly independent of bed size) while its target scales with the packed volume, so
      cv_new = cv * (target / purge) * (V_new / V_old), V ~ h_cms + h_al.
    Returns (params, slope used)."""
    last = history[-1]
    p = dict(last["params"])
    h_al = p["h_al"] if water_ok(last) else round(p["h_al"] * 1.3, 4)
    good = [r for r in history if abs(r["purge_step_mol"] / r["purge_target_mol"] - 1) < 0.05
            and abs(r["params"]["h_al"] - h_al) < 1e-6]
    slope = default_slope
    if len(good) >= 2:
        # the two good points closest to the target bracket it best
        g2 = sorted(good, key=lambda r: abs(math.log(r["O2_ppm"] / O2_TARGET)))[:2]
        if abs(g2[0]["params"]["h_cms"] - g2[1]["params"]["h_cms"]) > 1e-4:
            slope = ((math.log(g2[0]["O2_ppm"]) - math.log(g2[1]["O2_ppm"]))
                     / (g2[0]["params"]["h_cms"] - g2[1]["params"]["h_cms"]))
    slope = min(slope, -1.0)
    # step from the good point nearest the target if there is one (its purge is right), else the last
    base = min(good, key=lambda r: abs(math.log(r["O2_ppm"] / O2_TARGET))) if good else last
    h_new = base["params"]["h_cms"] + (math.log(O2_TARGET) - math.log(base["O2_ppm"])) / slope
    h_new = round(min(max(h_new, 0.5 * base["params"]["h_cms"]), 1.6 * base["params"]["h_cms"]), 4)
    v_old = last["params"]["h_cms"] + last["params"]["h_al"]
    v_new = h_new + h_al
    cv = p["cv_purge"] * last["purge_target_mol"] / last["purge_step_mol"] * v_new / v_old
    return dict(p, h_cms=h_new, h_al=h_al, cv_purge=cv), slope
