"""Fit the sorbent inputs from the literature data (env viz). Writes fits.json and fits.png.

Every number written here is a literature value, a fit to literature data, a value
calculated from them, or an assumption with its reason (docs/INPUTS.md lists them all):

CMS (Shirasagi MSC CT-350), 293.15 K
  O2, N2    single-site Langmuir fitted to Marcinek's Sips isotherms (M21d Tab. 3.3-1; reference
            temperature 293.15 K, checked against the ISODB data of the same CMS) over 0.02-10 bar
  Ar        = O2's isotherm (assumed: no ambient-temperature Ar data on CMS was reachable; Ar and O2
            have nearly equal polarizability and similar isotherms on carbons)
  CO2       single-site Langmuir fitted to ISODB 10.1007/s10450-016-9821-x (CT-350, 293 K)
  H2O       Langmuir fitted to ISODB 10.1021/la061140a (CMS, 293 K) at RH <= 0.22 only
  kinetics  LDF with Darken factor, k_i0 fitted so that a CMS pellet under a square-wave air
            pressure swing (8.0 / 1.0 bar, 59 s each) follows Marcinek's QDF + Darken + Arrhenius
            model (M21d Tab. 3.5-1, Eq. 3.5-6) at 293.15 K; Ar = N2 (assumed); CO2, H2O assumed
Activated alumina (F-200), 293.15 K
  H2O       dual-site Langmuir in relative humidity fitted to ISODB 10.1021/la901107s (303 K) at
            RH <= 0.5, moved to 293.15 K assuming loading depends on RH only
  CO2       single-site Langmuir fitted at 273 and 303 K (ISODB 10.1007/s10450-017-9907-0),
            b interpolated in 1/T to 293.15 K; used far below the measured pressures (Henry region)
  N2        single-site Langmuir fitted to ISODB 10.1007/s10934-011-9494-5 (alumina, 313 K), b moved
            to 293.15 K with an assumed 15 kJ/mol heat of adsorption; O2, Ar = N2 (assumed)
  kinetics  LDF from macropore diffusion (Glueckauf), calculated from assumed bead data

Run: conda run -n viz python -I fit_inputs.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "isodb"
LOCAL_ISODB = DATA  # this record was copied from psa-sim data/isodb/records so the repo stands alone
R = 8.314462618
T = 293.15
SP = ("N2", "O2", "Ar", "CO2", "H2O")


def psat_if97(Tk):
    n = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2, 0.12020824702470e5, -0.32325550322333e7,
         0.14915108613530e2, -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849, 0.65017534844798e3]
    th = Tk + n[8] / (Tk - n[9])
    A, B, C = th * th + n[0] * th + n[1], n[2] * th * th + n[3] * th + n[4], n[5] * th * th + n[6] * th + n[7]
    return (2 * C / (-B + math.sqrt(B * B - 4 * A * C))) ** 4 * 1e6  # Pa


def isodb(fn, folder=DATA, pmin=0.0, pmax=1e9, drop=()):
    d = json.load(open(folder / f"{fn}.json"))
    pts = []
    for i, p in enumerate(sorted(d["isotherm_data"], key=lambda r: r["pressure"])):
        q = p["total_adsorption"] if p.get("total_adsorption") is not None else p["species_data"][0]["adsorption"]
        if pmin <= p["pressure"] <= pmax and i not in drop:
            pts.append((p["pressure"], q))
    P, Q = np.array(pts).T
    if d["adsorptionUnits"] == "mol/g":
        Q = Q * 1e3
    assert d["pressureUnits"] == "bar" and d["adsorptionUnits"] in ("mmol/g", "mol/g"), fn
    return P, Q, d["temperature"], d["DOI"]


def langmuir(p, qs, b):
    return qs * b * p / (1 + b * p)


def fit_langmuir(p, q, x0=(3.0, 0.5)):
    r = least_squares(lambda v: (langmuir(p, *v) - q) / q, x0, bounds=([1e-6, 1e-8], [100, 1e5]))
    rel = np.abs(langmuir(p, *r.x) - q) / q
    return [float(v) for v in r.x], float(rel.max()), float(np.sqrt(np.mean(rel**2)))


# ---------------------------------------------------------------- CMS O2, N2 from Sips
SIPS = {  # M21d Tab. 3.3-1 (w_s0 kmol/kg -> mol/kg), T0 = 293.15 K
    "O2": dict(ws=3.384, b=9.436e-2, n=1.120),
    "N2": dict(ws=2.707, b=1.205e-1, n=1.185),
}


def sips(p, ws, b, n):
    x = (b * p) ** (1 / n)
    return ws * x / (1 + x)


out = {"T_K": T, "species": SP}
pg = np.logspace(np.log10(0.005), 1, 90)


def dual(p, q1, b1, q2, b2):
    return langmuir(p, q1, b1) + langmuir(p, q2, b2)


def fit_dual(p, q, x0):
    r = least_squares(lambda v: (dual(p, *v) - q) / q, x0, bounds=([1e-4, 1e-6, 1e-4, 1e-6], [50, 1e5, 50, 1e5]))
    v = [float(x) for x in r.x]
    if v[1] < v[3]:  # site 1 = the stronger site
        v = v[2:] + v[:2]
    rel = np.abs(dual(p, *v) - q) / q
    return v, float(rel.max()), float(np.sqrt(np.mean(rel**2)))


cms = {}
for s in ("O2", "N2"):
    (q1, b1, q2, b2), emax, erms = fit_dual(pg, sips(pg, **SIPS[s]), (0.3, 3.0, 3.0, 0.05))
    cms[s] = dict(qs=[q1, q2], b=[b1, b2], fit_max_rel_err_vs_sips=emax, fit_rms_rel_err=erms,
                  source="dual-site Langmuir fit to the Sips isotherm of M21d Tab. 3.3-1 at 293.15 K, 0.005-10 bar "
                         "(a single site misses the Sips curve by up to 20 % at 6-10 bar)")
# check against the ISODB data of the same material (N2 293 K, O2 298 K)
chk = {}
for s, fn in (("N2", "10.1002cite.201300063.Isotherm1"), ("O2", "10.1002cite.201300063.Isotherm4")):
    P, Q, Tm, doi = isodb(fn, pmax=10.5)
    chk[s] = dict(doi=doi, T=Tm, max_rel_dev_fit_vs_data=float(np.max(np.abs(dual(P, cms[s]["qs"][0], cms[s]["b"][0], cms[s]["qs"][1], cms[s]["b"][1]) - Q) / Q)))
cms["check_vs_isodb"] = chk
cms["Ar"] = dict(qs=cms["O2"]["qs"], b=cms["O2"]["b"], source="assumed equal to O2 (no ambient-T Ar data on CMS reachable)")
P1, Q1, _, doi1 = isodb("10.1007s104500169821x.Isotherm20", pmin=0.04)
P2, Q2, _, _ = isodb("10.1007s104500169821x.Isotherm41", pmin=0.2)
Pc, Qc = np.concatenate([P1, P2]), np.concatenate([Q1, Q2])
(qs, b), emax, erms = fit_langmuir(Pc, Qc, (4.0, 1.0))
cms["CO2"] = dict(qs=qs, b=b, fit_max_rel_err=emax, fit_rms_rel_err=erms, doi=doi1, p_range_bar=[float(Pc.min()), float(Pc.max())],
                  source="Langmuir fit to ISODB CT-350 293 K (Isotherm20 p >= 0.04 bar, Isotherm41 p >= 0.2 bar; "
                         "the scattered low-pressure points look unequilibrated)", extrapolated_below_bar=float(Pc.min()))
Pw, Qw, _, doiw = isodb("10.1021la061140a.Isotherm1", pmax=0.0052)
qs_w = 7.1  # mmol/g: highest measured loading of that isotherm (pore filling), used as capacity
bw = float(np.sum(Qw * Pw) / np.sum(Pw * Pw) / qs_w)
cms["H2O"] = dict(qs=qs_w, b=bw, doi=doiw, rh_max=float(Pw.max() * 1e5 / psat_if97(T)),
                  max_rel_err=float(np.max(np.abs(langmuir(Pw, qs_w, bw) - Qw) / Qw)),
                  source="Henry slope of ISODB la061140a Isotherm1 (CMS, 293 K) at RH <= 0.22, capacity 7.1 mmol/g "
                         "(its highest loading); the pore-filling rise above RH ~0.3 is not represented")
out["cms_isotherm"] = cms

# ---------------------------------------------------------------- alumina
# The data start at RH 0.043, where the strong site is already nearly full, so its constant is not
# constrained above ~100 (larger values fit about as well: max error 4.0 % at 3680 vs 6.0 % at 100);
# capping it avoids needless stiffness in the bed model.
B_RH_CAP = 100.0
Pa, Qa, Ta, doia = isodb("10.1021la901107s.Isotherm4")
RH = Pa * 1e5 / psat_if97(Ta)
m = RH <= 0.5
r = least_squares(lambda v: (langmuir(RH[m], v[0], v[1]) + langmuir(RH[m], v[2], v[3]) - Qa[m]) / Qa[m],
                  (3.0, 60.0, 30.0, 1.0), bounds=([0.1, 1, 0.1, 1e-3], [50, B_RH_CAP, 500, 50]))
q1, b1, q2, b2 = (float(v) for v in r.x)
rel = np.abs(langmuir(RH[m], q1, b1) + langmuir(RH[m], q2, b2) - Qa[m]) / Qa[m]
ps293 = psat_if97(T) / 1e5
alu = {"H2O": dict(qs=[q1, q2], b_rh=[b1, b2], b=[b1 / ps293, b2 / ps293], fit_max_rel_err=float(rel.max()),
                   rh_points=RH[m].round(4).tolist(), doi=doia, T_data=Ta,
                   source="dual-site Langmuir in RH fitted to ISODB la901107s Isotherm4 (F-200, 303 K) at RH <= 0.5; "
                          "moved to 293.15 K assuming loading depends on RH only (b = b_RH / p_sat(293.15 K)); strong-site "
                          "b_RH capped at 100 (not constrained by the data, which start at RH 0.043)", b_rh_cap=B_RH_CAP)}
# CO2 on alumina at 273 / 303 K -> 293.15 K
co2 = {}
for fn in ("10.1007s1045001799070.Isotherm1", "10.1007s1045001799070.Isotherm2"):
    P, Q, Tm, doi = isodb(fn)
    co2[Tm] = (P, Q)


def dual_T(p, Tk, v):
    q1, b1, h1, q2, b2, h2 = v  # b at 293.15 K, heats in kJ/mol
    f = lambda bb, hh: bb * np.exp(hh * 1e3 / R * (1 / Tk - 1 / T))  # noqa: E731
    return langmuir(p, q1, f(b1, h1)) + langmuir(p, q2, f(b2, h2))


res = least_squares(lambda v: np.concatenate([(dual_T(P, Tm, v) - Q) / Q for Tm, (P, Q) in co2.items()]),
                    (0.4, 50.0, 30.0, 1.5, 1.0, 15.0), bounds=([1e-3, 1e-2, 0, 1e-3, 1e-4, 0], [10, 1e5, 80, 20, 1e3, 80]))
v = [float(x) for x in res.x]
rel = np.concatenate([np.abs(dual_T(P, Tm, v) - Q) / Q for Tm, (P, Q) in co2.items()])
fits = {Tm: v for Tm in co2}
alu["CO2"] = dict(qs=[v[0], v[3]], b=[v[1], v[4]], heats_kJ_mol=[v[2], v[5]], fit_max_rel_err=float(rel.max()),
                  doi=doi, extrapolated_below_bar=0.0369,
                  source="dual-site Langmuir with per-site heats fitted jointly to 273 and 303 K (ISODB "
                         "10.1007/s10450-017-9907-0), evaluated at 293.15 K; feed CO2 (3.6 mbar) is below the measured range")
Pn, Qn, Tn, doin = isodb("10.1007s1093401194945.Isotherm3", folder=LOCAL_ISODB)
(qn, bn), en, _ = fit_langmuir(Pn, Qn, (2.0, 0.1))
bn293 = bn * math.exp(15000.0 / R * (1 / T - 1 / Tn))
alu["N2"] = dict(qs=qn, b=bn293, b_313=bn, fit_max_rel_err=en, doi=doin,
                 source="Langmuir fit to ISODB 10.1007/s10934-011-9494-5 (alumina, 313 K); b moved to 293.15 K with an "
                        "assumed 15 kJ/mol heat of adsorption")
alu["O2"] = dict(qs=qn, b=bn293, source="assumed equal to N2")
alu["Ar"] = dict(qs=qn, b=bn293, source="assumed equal to N2")
out["alumina_isotherm"] = alu

# ---------------------------------------------------------------- CMS kinetics (cyclic swing fit)
KIN = {"O2": (1.7534, 1622.33e-3), "N2": (118.8268, 25098.55e-3)}  # A/rp^2 (1/s), Ea (kJ/mol); M21d Tab. 3.5-1
k_qdf = {s: a * math.exp(-ea * 1e3 / (R * T)) for s, (a, ea) in KIN.items()}
y_air = {"O2": 0.2095, "N2": 0.7905}  # Ar lumped with N2 for this pellet test
iso = {s: (np.array(cms[s]["qs"]), np.array(cms[s]["b"])) for s in ("O2", "N2")}
P_H, P_L, t_half = 8.0, 1.0, 59.0


def qstar(P):
    bp = {s: iso[s][1] * y_air[s] * P for s in iso}  # per site
    den = 1 + sum(bp.values())
    return np.array([float(np.sum(iso[s][0] * bp[s] / den)) for s in ("O2", "N2")])


def run_swing(model, k0, n_cyc=40, nt=119):
    qsv = np.array([iso["O2"][0].sum(), iso["N2"][0].sum()])
    nsv = np.array([SIPS["O2"]["n"], SIPS["N2"]["n"]])
    wsv = np.array([SIPS["O2"]["ws"], SIPS["N2"]["ws"]])

    def rhs(t, w, P):
        ws_ = qstar(P)
        if model == "qdf":  # Marcinek Eq. 3.5-6: k (w/p dp/dw)_Sips (w*^2 - w^2)/(2w)
            th = np.clip(w / wsv, 0, 0.95)
            return np.array([k_qdf["O2"], k_qdf["N2"]]) * nsv / (1 - th) * (ws_**2 - w**2) / (2 * np.maximum(w, 1e-9))
        th = min(np.sum(w / qsv), 0.95)
        return np.asarray(k0) / (1 - th) * (ws_ - w)

    w = qstar(P_L)
    traj = []
    for c in range(n_cyc):
        for P in (P_H, P_L):
            s = solve_ivp(rhs, (0, t_half), w, args=(P,), method="LSODA", rtol=1e-8, atol=1e-10,
                          t_eval=np.linspace(0, t_half, nt))
            w = s.y[:, -1]
            if c == n_cyc - 1:
                traj.append(s.y)
    return np.concatenate(traj, axis=1)


ref = run_swing("qdf", None)
amp_ref = ref.max(1) - ref.min(1)
fit = least_squares(lambda lk: ((run_swing("ldf", np.exp(lk)) - ref) / amp_ref[:, None]).ravel(),
                    np.log([1.0, 0.01]), x_scale=1.0)
k0 = np.exp(fit.x)
ldf = run_swing("ldf", k0)
amp = ldf.max(1) - ldf.min(1)
out["cms_kinetics"] = dict(
    model="LDF with Darken factor 1/(1-theta_total)",
    k0_per_s={"O2": float(k0[0]), "N2": float(k0[1]), "Ar": float(k0[1])},
    k_qdf_293K_zero_loading_per_s=k_qdf,
    swing_test=dict(P_high_bar=P_H, P_low_bar=P_L, half_s=t_half, y_air=y_air),
    swing_amplitude_mol_kg={"qdf": amp_ref.tolist(), "ldf_fit": amp.tolist()},
    swing_traj_max_rel_err=float(np.max(np.abs(ldf - ref) / amp_ref[:, None])),
    k0_CO2=0.1 * float(k0[0]), k0_H2O=0.01,
    source="k0 fitted so that the LDF+Darken pellet follows Marcinek's QDF+Darken+Arrhenius model (M21d Tab. 3.5-1) "
           "through a square-wave air swing; Ar = N2 (assumed: Ar/N2 rate ratios on CMS in the literature lie on both "
           "sides of 1); CO2 = 0.1 x O2 (assumed: a small linear molecule, reported on CMS as far faster than N2 but "
           "slower than O2; no rate constant for CT-350 was reachable); H2O 0.01 /s (assumed; water stays in the alumina, G1f)")

# ---------------------------------------------------------------- alumina kinetics (Glueckauf)
# Assumed (F-200-like beads): 14x28 mesh (mean diameter ~1.2 mm), the small beads used in fast-cycle PSA
# pretreatment layers; activated alumina is bidisperse, so uptake is controlled by molecular diffusion in the
# macropores (tortuosity 3). A first assumption (7x12 mesh, 2.4 mm, Knudsen diffusion in the 5.7 nm mesopores)
# gave a water LDF of 6.6e-5 /s: the layer could not regenerate within a 120 s cycle, saturated, and passed the
# water into the CMS (docs/LOG.md).
BEAD = dict(radius=0.6e-3, porosity=0.55, density=1250.0, pore_d=5.7e-9, tortuosity=3.0, transport="macropore")
M = {"N2": 0.028, "O2": 0.032, "Ar": 0.040, "CO2": 0.044, "H2O": 0.018}
D_air_1bar = {"N2": 2.0e-5, "O2": 2.0e-5, "Ar": 1.9e-5, "CO2": 1.6e-5, "H2O": 2.4e-5}  # m^2/s at 293 K, 1 bar (assumed)
P_mid = 4.5  # bar, cycle-average pressure for the molecular diffusivity
y_feed_wet = {"N2": 0.78078, "O2": 0.20946, "Ar": 0.00934, "CO2": 0.00042, "H2O": 0.000915}
P_feed = 8.51325
k_alu, slopes = {}, {}
for s in SP:
    Dm = D_air_1bar[s] / P_mid
    De = BEAD["porosity"] / BEAD["tortuosity"] * Dm  # macropore (molecular) diffusion controls
    a = alu[s]
    p = y_feed_wet[s] * P_feed
    qs_, b_ = np.atleast_1d(a["qs"]), np.atleast_1d(a["b"])
    q = float(np.sum(qs_ * b_ * p / (1 + b_ * p)))
    c = p * 1e5 / (R * T)
    slope = q / c  # m^3/kg chord from a clean pellet to the feed
    slopes[s] = slope
    k_alu[s] = 15 * De / (BEAD["radius"] ** 2 * (BEAD["porosity"] + BEAD["density"] * slope))
out["alumina_kinetics"] = dict(k_per_s=k_alu, chord_slope_m3_kg=slopes, bead=BEAD, P_mid_bar=P_mid,
                               source="Glueckauf LDF from macropore (molecular) diffusion, D_e = eps_p D_m / tau, chord from a "
                                      "clean bead to the feed partial pressure; bead data assumed (F-200-like 14x28 mesh)")
def site_arrays(block, site_of):
    qs_a, b_a = np.zeros((5, 2)), np.zeros((5, 2))
    for i, s in enumerate(SP):
        q_, b_ = np.atleast_1d(block[s]["qs"]), np.atleast_1d(block[s]["b"])
        if q_.size == 2:
            qs_a[i], b_a[i] = q_, b_
        else:
            qs_a[i, site_of[s]], b_a[i, site_of[s]] = q_[0], b_[0]
    return qs_a.tolist(), b_a.tolist()


kin = out["cms_kinetics"]
qs_c, b_c = site_arrays(cms, {"CO2": 1, "H2O": 1})
qs_a, b_a = site_arrays(alu, {"N2": 1, "O2": 1, "Ar": 1})
out["solver_params"] = dict(
    order=SP, sites="site 0 = strong site; single-site species placed as noted",
    cms=dict(qs=qs_c, b=b_c, k0=[kin["k0_per_s"]["N2"], kin["k0_per_s"]["O2"], kin["k0_per_s"]["Ar"], kin["k0_CO2"], kin["k0_H2O"]],
             darken=True, single_site_placement="CO2, H2O on site 1 (the main site; site 0 is a tiny strong site of N2/O2/Ar whose large b would crowd them out)"),
    alumina=dict(qs=qs_a, b=b_a, k0=[k_alu[s] for s in SP], darken=False,
                 single_site_placement="N2, O2, Ar on site 1 (not displaced by water's strong site); CO2 and H2O dual-site"))
(HERE / "fits.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "species"}, indent=1)[:6000])

# ---------------------------------------------------------------- figure
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
a = ax[0, 0]
for s, col in (("O2", "#eb6834"), ("N2", "#1baf7a")):
    a.plot(pg, sips(pg, **SIPS[s]), color=col, lw=1, ls="--", label=f"{s} Sips (M21d)")
    a.plot(pg, dual(pg, cms[s]["qs"][0], cms[s]["b"][0], cms[s]["qs"][1], cms[s]["b"][1]), color=col, lw=2, label=f"{s} dual-site fit")
for s, fn, col in (("N2", "10.1002cite.201300063.Isotherm1", "#1baf7a"), ("O2", "10.1002cite.201300063.Isotherm4", "#eb6834")):
    P, Q, Tm, _ = isodb(fn, pmax=10.5)
    a.plot(P, Q, "o", color=col, ms=4, label=f"{s} ISODB {Tm} K")
a.set(title="CMS: O2, N2 (293 K)", xlabel="p (bar)", ylabel="q (mol/kg)")
a.legend(fontsize=8)
a = ax[0, 1]
a.plot(Pc, Qc, "o", color="#52514e", ms=4)
pp = np.logspace(-3, np.log10(20), 100)
a.plot(pp, langmuir(pp, cms["CO2"]["qs"], cms["CO2"]["b"]), color="#0b0b0b", lw=2)
a.axvline(0.00042 * P_feed, color="#898781", ls=":", lw=1)
a.set(xscale="log", title="CMS: CO2 (293 K); dotted = feed p_CO2", xlabel="p (bar)", ylabel="q (mol/kg)")
a = ax[0, 2]
rr = np.linspace(0, 0.9, 100)
a.plot(RH, Qa, "o", color="#2a78d6", ms=5, label="ISODB F-200, 303 K")
a.plot(rr, langmuir(rr, q1, b1) + langmuir(rr, q2, b2), color="#2a78d6", lw=2, label="dual-site fit (RH <= 0.5)")
a.axvline(0.000915 * P_feed * 1e5 / psat_if97(T), color="#898781", ls=":", lw=1)
a.set(title="Alumina: H2O vs RH; dotted = feed RH at 20 C", xlabel="RH", ylabel="q (mol/kg)", ylim=(0, 21))
a.legend(fontsize=8)
a = ax[1, 0]
for fn, col in (("10.1007s1045001799070.Isotherm1", "#2a78d6"), ("10.1007s1045001799070.Isotherm2", "#eb6834")):
    P, Q, Tm, _ = isodb(fn)
    a.plot(P, Q, "o", color=col, ms=4, label=f"{Tm} K")
    a.plot(pp[pp < 1.2], dual_T(pp[pp < 1.2], Tm, v), color=col, lw=1.5)
a.plot(pp[pp < 1.2], dual_T(pp[pp < 1.2], T, v), color="#0b0b0b", lw=2, label="293 K (joint fit)")
a.axvline(0.00042 * P_feed, color="#898781", ls=":", lw=1)
a.set(xscale="log", title="Alumina: CO2", xlabel="p (bar)", ylabel="q (mol/kg)")
a.legend(fontsize=8)
a = ax[1, 1]
tt = np.linspace(0, 2 * t_half, ref.shape[1])
for i, (s, col) in enumerate((("O2", "#eb6834"), ("N2", "#1baf7a"))):
    a.plot(tt, ref[i], color=col, lw=1, ls="--", label=f"{s} QDF+Darken (M21d)")
    a.plot(tt, ldf[i], color=col, lw=2, label=f"{s} LDF+Darken fit")
a.set(title="CMS pellet, 8/1 bar air swing (periodic)", xlabel="t (s)", ylabel="q (mol/kg)")
a.legend(fontsize=8)
a = ax[1, 2]
a.plot(Pn, Qn, "o", color="#1baf7a", ms=4, label="N2 on alumina, 313 K (ISODB)")
a.plot(pp[pp < 6], langmuir(pp[pp < 6], qn, bn), color="#1baf7a", lw=1.5, label="fit 313 K")
a.plot(pp[pp < 6], langmuir(pp[pp < 6], qn, bn293), color="#0b0b0b", lw=2, label="moved to 293 K")
a.set(title="Alumina: N2 (O2, Ar assumed equal)", xlabel="p (bar)", ylabel="q (mol/kg)")
a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "fits.png", dpi=110)
