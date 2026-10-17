"""Correctness and optimality checks for the linear programming solver."""

import numpy as np
import pytest

from src.floorplan import P_max, T_amb
from src.optimizer import solve_lp
from src.power_model import block_powers
from src.thermal_solver import build_A, build_B, solve_from_regions


@pytest.fixture(scope="module")
def system_matrices():
    """Build and cache system matrices A and B for LP tests."""
    A = build_A()
    B = build_B()
    return A, B


def test_lp_success(system_matrices):
    """Assert LP returns success=True for P_max=5.0."""
    A, B = system_matrices
    res = solve_lp(A, B, P_max=5.0)
    assert res["success"] is True
    assert "optimal" in res["message"].lower() or "terminated successfully" in res["message"].lower()


def test_lp_budget_equality(system_matrices):
    """Assert total power sum(q_j) == 5.0 within 1e-6 tolerance."""
    A, B = system_matrices
    res = solve_lp(A, B, P_max=5.0)
    total_q = float(np.sum(res["q"]))
    assert np.isclose(total_q, 5.0, atol=1e-6), f"Total power {total_q} != 5.0 W"


def test_lp_not_worse_than_uniform(system_matrices):
    """Assert LP peak temperature <= uniform peak temperature + 1e-6."""
    A, B = system_matrices

    # Scenario B (asymmetric: cores 0 & 1 active)
    raw_p = block_powers(active_cores=(0, 1))
    raw_q = np.array([raw_p["core0"], raw_p["core1"], raw_p["core2"], raw_p["core3"], raw_p["l2"]])
    q_uniform = raw_q * (5.0 / np.sum(raw_q))

    theta_u = solve_from_regions(q_uniform, A, B)
    peak_uniform = float(T_amb + np.max(theta_u))

    res_lp = solve_lp(A, B, P_max=5.0)
    peak_lp = float(T_amb + res_lp["t_peak"])

    assert peak_lp <= peak_uniform + 1e-6, f"LP peak ({peak_lp}) > uniform peak ({peak_uniform})"


def test_lp_symmetric_scenario_a(system_matrices):
    """Assert LP preserves four-core symmetry in Scenario A (cores 0..3 receive uniform power)."""
    A, B = system_matrices
    res_lp = solve_lp(A, B, P_max=5.0)
    q = res_lp["q"]

    # In symmetric layout, with q_L2 >= 0.30 W, the remaining 4.70 W is distributed
    # identically across the 4 peripheral cores: 4.70 / 4 = 1.175 W each.
    q_cores = q[:4]
    mean_core_power = float(np.mean(q_cores))
    assert np.allclose(q_cores, mean_core_power, atol=1e-3), (
        f"Core powers are not uniform: {q_cores}"
    )
    assert np.isclose(mean_core_power, 1.175, atol=1e-3)


def test_lp_l2_lower_bound(system_matrices):
    """Assert q_L2 >= 0.30 - 1e-6 and each q_core >= 0.10 - 1e-6 for both scenarios with P_max = 5.0."""
    A, B = system_matrices

    # Solve LP with P_max = 5.0
    res = solve_lp(A, B, P_max=5.0)
    q = res["q"]
    q_cores = q[:4]
    q_l2 = q[4]

    assert q_l2 >= 0.30 - 1e-6, f"L2 cache power {q_l2:.4f} W < 0.30 W"
    for idx, qc in enumerate(q_cores):
        assert qc >= 0.10 - 1e-6, f"Core {idx} power {qc:.4f} W < 0.10 W"

