"""Linear programming optimizer for thermal-aware power budget redistribution.

This module formulates and solves the minimax temperature optimization problem
using an epigraph reformulation, redistributing a fixed power budget P_max across
the five functional blocks to minimize peak die temperature.

Optimization problem (epigraph form):
    minimize   t
    subject to A θ - B q = 0
               θ_i - t ≤ 0,         i = 1 .. N²
              -θ_i - t ≤ 0,         i = 1 .. N²
               Σ_j q_j = P_max
               q_core ≥ 0.10 W,     j = 1 .. 4  (prevents core shutdown)
               q_L2   ≥ 0.30 W                  (realistic architectural lower bound)
               t ≥ 0

Decision vector:
    x = [q (5), θ (1600), t (1)]  -->  1606 variables
"""

from typing import Any, Dict
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .floorplan import N, P_max as DEFAULT_P_MAX


def solve_lp(
    A: sp.spmatrix,
    B: np.ndarray,
    P_max: float = DEFAULT_P_MAX,
) -> Dict[str, Any]:
    """Solve the minimax temperature linear program via scipy.optimize.linprog (HiGHS).

    Args:
        A: (N², N²) sparse finite-difference Laplacian conductance matrix (W/K).
        B: (N², 5) power distribution matrix mapping block powers to nodes.
        P_max: Total power budget in Watts (strict equality constraint).

    Returns:
        Dict[str, Any] containing:
            - 'success': (bool) True if optimizer converged to optimality.
            - 'q': (np.ndarray) Optimal 5-element block powers [q0, q1, q2, q3, q4] (W).
            - 'theta': (np.ndarray) Resulting (N, N) temperature rise field θ (K).
            - 't_peak': (float) Peak temperature rise max(θ) across the die (K).
            - 'message': (str) Solver status message.

    Raises:
        AssertionError: If res.success is False.
    """
    n_nodes = N * N
    num_blocks = 5
    n_vars = num_blocks + n_nodes + 1  # 5 + 1600 + 1 = 1606

    # Objective: minimize epigraph variable t
    # c = [0]*5 + [0]*1600 + [1]
    c = np.zeros(n_vars, dtype=float)
    c[-1] = 1.0

    # Convert A to CSR if not already
    A_csr = A.tocsr() if isinstance(A, sp.spmatrix) else sp.csr_matrix(A)
    B_csr = sp.csr_matrix(B)

    # 1. Equality constraints:
    # Row 0..1599: A θ - B q = 0  --> [-B, A, 0] [q, θ, t]^T = 0
    zeros_eq_col = sp.csr_matrix((n_nodes, 1), dtype=float)
    eq_pde = sp.hstack([-B_csr, A_csr, zeros_eq_col], format="csr")

    # Row 1600: Σ q_j = P_max --> [1, 1, 1, 1, 1, 0...0, 0]
    eq_budget = sp.csr_matrix(
        ([1.0] * num_blocks, ([0] * num_blocks, list(range(num_blocks)))),
        shape=(1, n_vars),
        dtype=float,
    )
    A_eq = sp.vstack([eq_pde, eq_budget], format="csr")

    b_eq = np.zeros(n_nodes + 1, dtype=float)
    b_eq[-1] = float(P_max)

    # 2. Inequality constraints: -t ≤ θ_i ≤ t
    # Row 0..1599:    θ_i - t ≤ 0   --> [0,  I, -1]
    # Row 1600..3199: -θ_i - t ≤ 0  --> [0, -I, -1]
    zeros_ub = sp.csr_matrix((n_nodes, num_blocks), dtype=float)
    I_nodes = sp.eye(n_nodes, format="csr", dtype=float)
    minus_ones = sp.csr_matrix(-np.ones((n_nodes, 1), dtype=float))

    ub_upper = sp.hstack([zeros_ub, I_nodes, minus_ones], format="csr")
    ub_lower = sp.hstack([zeros_ub, -I_nodes, minus_ones], format="csr")
    A_ub = sp.vstack([ub_upper, ub_lower], format="csr")

    b_ub = np.zeros(2 * n_nodes, dtype=float)

    # 3. Variable bounds:
    # q_core_j ≥ 0.10 W (cores 0..3), q_L2 ≥ 0.30 W (L2 cache), θ_i unconstrained, t ≥ 0
    bounds = (
        [(0.10, None)] * 4
        + [(0.30, None)]
        + [(None, None)] * n_nodes
        + [(0.0, None)]
    )

    # Solve using HiGHS dual simplex / interior point
    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    print(f"LP Status: success={res.success}, message='{res.message}'")
    assert res.success, f"Linear programming failed: {res.message}"

    q_opt = res.x[:num_blocks]
    theta_opt = res.x[num_blocks : num_blocks + n_nodes].reshape((N, N))
    t_peak = float(res.x[-1])

    return {
        "success": bool(res.success),
        "q": q_opt,
        "theta": theta_opt,
        "t_peak": t_peak,
        "message": str(res.message),
    }
