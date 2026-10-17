"""Sanity checks and physical verification of the finite-difference thermal solver."""

import numpy as np
import scipy.sparse.linalg as spla
import pytest

from src.floorplan import L, N, T_amb, h
from src.modal_verification import modal_Rth
from src.power_model import block_powers
from src.thermal_solver import build_A, build_B, solve_from_regions
from src.visualize import _build_padded_grid


def test_matrix_a_symmetric():
    """Assert matrix A is mathematically symmetric."""
    A = build_A()
    diff = A - A.T
    assert diff.nnz == 0 or np.allclose(diff.data, 0.0, atol=1e-12)


def test_matrix_a_positive_definite():
    """Assert matrix A is positive definite by checking leading principal submatrix eigenvalues."""
    A = build_A()
    sub_A = A[:100, :100].toarray()
    eigvals = np.linalg.eigvalsh(sub_A)
    assert np.all(eigvals > 0.0), f"Found non-positive eigenvalues: {eigvals[eigvals <= 0]}"


def test_point_source_matches_modal():
    """Assert solving with a point source at (0.25L, 0.25L) matches modal_Rth to 1e-10."""
    A = build_A()
    x0 = 0.25 * L
    y0 = 0.25 * L

    i0 = int(round(x0 / h)) - 1
    j0 = int(round(y0 / h)) - 1
    node_x = (i0 + 1) * h
    node_y = (j0 + 1) * h

    e_source = np.zeros(N * N, dtype=float)
    flat_idx = i0 * N + j0
    e_source[flat_idx] = 1.0

    theta_sparse = spla.spsolve(A, e_source)
    r_sparse = float(theta_sparse[flat_idx])

    r_modal = modal_Rth(node_x, node_y, node_x, node_y, n_modes=N, discrete=True)

    rel_diff = abs(r_modal - r_sparse) / r_sparse
    assert rel_diff < 1e-10, f"Relative difference {rel_diff:.3e} exceeds 1e-10 tolerance"


def test_matrix_b_shape_and_conservation():
    """Assert matrix B has proper dimensions and each column sums to 1.0."""
    B = build_B()
    assert B.shape == (N * N, 5)
    col_sums = np.sum(B, axis=0)
    assert np.allclose(col_sums, 1.0, atol=1e-12)


def test_3d_surface_max_matches_2d_heatmap():
    """Assert the 3D surface's maximum Z value matches the 2D heatmap's maximum temperature within 1e-6 K."""
    A = build_A()
    B = build_B()

    # Test with Scenario B power distribution
    raw_p = block_powers(active_cores=(0, 1))
    raw_q = np.array([raw_p["core0"], raw_p["core1"], raw_p["core2"], raw_p["core3"], raw_p["l2"]])
    q_uniform = raw_q * (5.0 / np.sum(raw_q))

    theta = solve_from_regions(q_uniform, A, B)

    # 2D heatmap maximum temperature
    max_2d = float(T_amb + np.max(theta))

    # 3D surface Z grid
    _, _, T_3d = _build_padded_grid(theta)
    max_3d = float(np.max(T_3d))

    diff = abs(max_3d - max_2d)
    assert diff < 1e-6, f"3D max Z ({max_3d:.8f} K) does not match 2D max ({max_2d:.8f} K), diff={diff:.3e}"
