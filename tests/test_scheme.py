"""The van Albada option keeps the kernel conservative and reduces to upwind on flat profiles."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from psan2viz.cases import bed_grid, load_solver_params  # noqa: E402


def _state(grid, rng):
    n, N = grid.n, grid.N
    C = 300.0 + 10 * rng.random()
    y = rng.random((n, N)) + 0.1
    y /= y.sum(0)
    c = (y * C)[:, :, None]
    q = grid.equilibrium(c) * (0.5 + rng.random((n, N, 1)))
    return c, q, c.sum(0)


def _ends(grid, c, Fb, Ft):
    yb, yt = c[:, 0] / c[:, 0].sum(0), c[:, -1] / c[:, -1].sum(0)
    return ("valves", np.array([Fb]), Fb * yb), ("valves", np.array([Ft]), Ft * yt)


def test_vanalbada_conserves_and_matches_upwind_on_flat_profiles():
    rng = np.random.default_rng(1)
    p = load_solver_params()
    g_up = bed_grid(0.6, 0.06, 0.05, 30, 6, p, "upwind")
    g_va = bed_grid(0.6, 0.06, 0.05, 30, 6, p, "vanalbada")
    c, q, C = _state(g_va, rng)
    bot, top = _ends(g_va, c, 0.05, -0.02)
    dc, dq, u, fb, ft = g_va.rates(c, q, C, bot, top)
    # conservation: d(inventory)/dt = flow in at the bottom - flow out at the top
    w_gas = (g_va.area * g_va.epsdz)[:, None]
    w_ads = (g_va.area * g_va.rho * g_va.dz)[:, None]
    dinv = (w_gas * dc).sum(1) + (w_ads * dq).sum(1)
    assert np.allclose(dinv, fb - ft, rtol=1e-10, atol=1e-14)
    # flat gas profile: van Albada == upwind
    cf = np.repeat(c[:, :1], g_va.N, axis=1)
    qf = np.repeat(q[:, :1], g_va.N, axis=1)
    b2, t2 = _ends(g_va, cf, 0.05, -0.02)
    r_va = g_va.rates(cf, qf, cf.sum(0), b2, t2)
    r_up = g_up.rates(cf, qf, cf.sum(0), b2, t2)
    assert np.allclose(r_va[0], r_up[0], rtol=1e-9, atol=1e-12)
