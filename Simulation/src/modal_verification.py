"""Analytical modal verification of the 5-point finite-difference thermal solver.

This module computes the thermal resistance R_th between points on a square die
using an eigenfunction expansion (modal double sine series) and verifies the
discrete sparse matrix solver against the analytical modal solution.

Eigenfunctions of -∇² on [0, L] × [0, L] with zero Dirichlet BC:
    φ_mn(x, y) = (2 / L) * sin(m * π * x / L) * sin(n * π * y / L)
    λ_mn = (π² / L²) * (m² + n²)

Thermal resistance:
    R_th(x, y; x0, y0) = Σ_{m,n ≥ 1} φ_mn(x0, y0) * φ_mn(x, y) / (k * t_d * λ_mn)
"""

from typing import Optional
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .floorplan import L, N, h, k, t_d


def modal_Rth(
    x0: float,
    y0: float,
    x: float,
    y: float,
    n_modes: int = 100,
    discrete: bool = True,
) -> float:
    """Compute the steady-state thermal resistance R_th via modal sine series.

    Vectorized using numpy outer products over m, n in [1, n_modes].

    Args:
        x0: Heat source x-coordinate in metres.
        y0: Heat source y-coordinate in metres.
        x: Observation point x-coordinate in metres.
        y: Observation point y-coordinate in metres.
        n_modes: Number of modes to sum along each spatial dimension.
        discrete: If True, uses the exact discrete eigenvalues and mode shapes
            of the 5-point Laplacian stencil on the N x N grid (matches matrix A
            to machine precision). If False, evaluates the continuous Laplacian
            eigenvalues λ_mn = π²(m² + n²) / L².

    Returns:
        float: Thermal resistance R_th in K/W.
    """
    if discrete:
        # Number of discrete modes cannot exceed grid resolution N
        actual_modes = min(n_modes, N)
        m = np.arange(1, actual_modes + 1)
        n = np.arange(1, actual_modes + 1)

        # Discrete sinusoidal basis functions on grid nodes
        # w_mn(i, j) = sqrt(2 / (N+1)) * sin(m*pi*x/L)
        phi_x_src = np.sin(m * np.pi * (x0 / L))
        phi_y_src = np.sin(n * np.pi * (y0 / L))
        phi_x_obs = np.sin(m * np.pi * (x / L))
        phi_y_obs = np.sin(n * np.pi * (y / L))

        Phi_src = np.outer(phi_x_src, phi_y_src) * (2.0 / (N + 1))
        Phi_obs = np.outer(phi_x_obs, phi_y_obs) * (2.0 / (N + 1))

        # Discrete eigenvalues of A / (k * t_d): 4 - 2*cos(m*pi/(N+1)) - 2*cos(n*pi/(N+1))
        lambda_m = 2.0 - 2.0 * np.cos(m * np.pi / (N + 1))
        lambda_n = 2.0 - 2.0 * np.cos(n * np.pi / (N + 1))
        Lambda = k * t_d * (lambda_m[:, None] + lambda_n[None, :])

        R_th = float(np.sum((Phi_src * Phi_obs) / Lambda))
    else:
        m = np.arange(1, n_modes + 1)
        n = np.arange(1, n_modes + 1)

        phi_x_src = np.sin(m * np.pi * x0 / L)
        phi_y_src = np.sin(n * np.pi * y0 / L)
        phi_x_obs = np.sin(m * np.pi * x / L)
        phi_y_obs = np.sin(n * np.pi * y / L)

        Phi_src = np.outer(phi_x_src, phi_y_src) * (2.0 / L)
        Phi_obs = np.outer(phi_x_obs, phi_y_obs) * (2.0 / L)

        lambda_mn = (np.pi**2 / L**2) * (m[:, None] ** 2 + n[None, :] ** 2)
        denom = k * t_d * lambda_mn

        R_th = float(np.sum((Phi_src * Phi_obs) / denom))

    return R_th


def verify_against_sparse(A: sp.spmatrix) -> float:
    """Verify sparse finite-difference matrix A against the analytical modal solution.

    Places a unit 1 W point source at (0.25 L, 0.25 L), solves using both
    scipy.sparse.linalg.spsolve and the modal expansion, prints a summary table,
    and asserts relative difference is strictly below 1e-10.

    Args:
        A: (N², N²) sparse matrix representing the 5-point discrete Laplacian.

    Returns:
        float: Relative difference |R_modal - R_sparse| / R_sparse.

    Raises:
        AssertionError: If relative difference exceeds 1e-10.
    """
    x0 = 0.25 * L
    y0 = 0.25 * L

    # Map (0.25 L, 0.25 L) to the closest interior grid node (0-indexed)
    i0 = int(round(x0 / h)) - 1
    j0 = int(round(y0 / h)) - 1
    node_x = (i0 + 1) * h
    node_y = (j0 + 1) * h

    # Unit point source vector
    e_source = np.zeros(N * N, dtype=float)
    flat_idx = i0 * N + j0
    e_source[flat_idx] = 1.0

    # Numerical solution via sparse direct solver
    theta_sparse = spla.spsolve(A, e_source)
    r_sparse = float(theta_sparse[flat_idx])

    # Modal solutions
    r_modal_discrete = modal_Rth(node_x, node_y, node_x, node_y, n_modes=N, discrete=True)
    r_modal_cont = modal_Rth(node_x, node_y, node_x, node_y, n_modes=100, discrete=False)

    rel_diff = abs(r_modal_discrete - r_sparse) / r_sparse

    print("=" * 68)
    print("MODAL VERIFICATION SUMMARY: POINT SOURCE AT (0.25 L, 0.25 L)")
    print("=" * 68)
    print(f"Grid Node:                  ({i0}, {j0})  [x = {node_x * 1e3:.3f} mm, y = {node_y * 1e3:.3f} mm]")
    print(f"Sparse Solver R_th:         {r_sparse:.9f} K/W")
    print(f"Discrete Modal R_th:        {r_modal_discrete:.9f} K/W")
    print(f"Continuous Modal (100 m.):  {r_modal_cont:.9f} K/W (approx due to point singularity)")
    print(f"Relative Difference:        {rel_diff:.4e}")
    print(f"Assertion (diff < 1e-10):   {'PASSED' if rel_diff < 1e-10 else 'FAILED'}")
    print("=" * 68)

    assert rel_diff < 1e-10, f"Modal verification failed: rel_diff={rel_diff:.3e} >= 1e-10"
    return rel_diff
