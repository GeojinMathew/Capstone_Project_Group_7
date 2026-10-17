"""Parametric sweep over total die power budget P_max.

This module evaluates both uniform power allocation and LP-optimal power
allocation across a range of total die power budgets (1 W to 8 W in 15 steps),
quantifying how peak die temperature scales with available power.
"""

from typing import Optional, Sequence, Tuple
import numpy as np

from .floorplan import T_amb
from .optimizer import solve_lp
from .power_model import block_powers
from .thermal_solver import build_A, build_B, solve_from_regions


def sweep_power_budget(
    active_cores: Sequence[int] = (0, 1),
    P_range: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep total power budget P_max and compare peak temperatures.

    For each budget level in P_range:
    1. Scale the baseline uniform/nominal power map to sum to P_max, solve the PDE,
       and record the peak temperature.
    2. Solve the linear program with budget P_max and record the optimized peak temperature.

    Args:
        active_cores: Sequence of active core indices (e.g. (0, 1) for Scenario B).
        P_range: Optional array of power budgets in Watts. Defaults to 15 evenly
            spaced values from 1.0 W to 8.0 W.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - P_array: (15,) array of power budget values (W).
            - peak_uniform: (15,) array of peak temperatures under uniform allocation (K).
            - peak_lp: (15,) array of peak temperatures under LP-optimal allocation (K).
    """
    if P_range is None:
        P_array = np.linspace(1.0, 8.0, 15)
    else:
        P_array = np.asarray(P_range, dtype=float)

    # Pre-build sparse system matrices once
    A = build_A()
    B = build_B()

    # Base block power values for given operating scenario
    raw_powers = block_powers(active_cores=active_cores)
    raw_q_vec = np.array([
        raw_powers["core0"],
        raw_powers["core1"],
        raw_powers["core2"],
        raw_powers["core3"],
        raw_powers["l2"],
    ], dtype=float)
    raw_sum = float(np.sum(raw_q_vec))

    peak_uniform_list = []
    peak_lp_list = []

    for p_budget in P_array:
        # 1. Scale baseline allocation to sum to current budget
        q_scaled = raw_q_vec * (p_budget / raw_sum)
        theta_u = solve_from_regions(q_scaled, A, B)
        t_peak_u = float(T_amb + np.max(theta_u))
        peak_uniform_list.append(t_peak_u)

        # 2. Run LP optimization with current budget
        lp_res = solve_lp(A, B, P_max=p_budget)
        t_peak_lp = float(T_amb + lp_res["t_peak"])
        peak_lp_list.append(t_peak_lp)

    peak_uniform = np.array(peak_uniform_list, dtype=float)
    peak_lp = np.array(peak_lp_list, dtype=float)

    return P_array, peak_uniform, peak_lp
