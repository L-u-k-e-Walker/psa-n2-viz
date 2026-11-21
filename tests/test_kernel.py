"""Fast checks of the shared kernel against the frozen psa-viz model (one cycle, coarse grid)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "model" / "regression")]
from psaviz_ref.cases import column_case  # noqa: E402
from psaviz_ref.column import ColumnModel  # noqa: E402
from g1c_regression import N2, O2, new_case  # noqa: E402
from psan2viz.column import ColumnN  # noqa: E402


def test_one_cycle_matches_psaviz():
    ref_case = column_case(n_cells=30, rtol=1e-9)
    m_ref = ColumnModel(ref_case)
    x_ref, recs_ref = m_ref.run_cycle(m_ref.initial_state())
    m_new = ColumnN(new_case(ref_case, 1e-9))
    x_new, recs_new = m_new.run_cycle(m_new.initial_state())
    c_ref, q_ref, cR_ref, _ = m_ref.split(x_ref)
    c_new, q_new, cR_new, _ = m_new.split(x_new)
    scale = ref_case.p_feed * 1e5 / m_ref.RT
    assert np.max(np.abs(c_ref[0] - c_new[O2])) / scale < 1e-6
    assert np.max(np.abs(c_ref[1] - c_new[N2])) / scale < 1e-6
    assert np.max(np.abs(q_ref[0] - q_new[O2])) < 1e-6
    assert np.max(np.abs(q_ref[1] - q_new[N2])) < 1e-6
    assert np.max(np.abs(cR_ref[1] - cR_new[N2])) / scale < 1e-6
    assert np.max(np.abs(c_new[2:])) / scale < 1e-15  # absent species stay zero (round-off only)
    assert max(np.max(np.abs(r["imbalance"])) for r in recs_new) < 1e-9
