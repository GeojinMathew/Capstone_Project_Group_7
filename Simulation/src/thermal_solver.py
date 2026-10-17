"""Finite-difference thermal solver for the steady-state heat equation.

This module sets up the discrete 5-point finite-difference Poisson operator A,
the power mapping matrix B, and computes temperature rise solutions (θ in Kelvin)
given nodal power maps or block power vectors.

Governing PDE (steady state):
    k * t_d * ∇²θ = -q''(x, y)
    θ = 0 on all four boundaries (Dirichlet BC, T = 300 K)

Discrete system:
    A θ = B q
where:
    A = k * t_d * (I ⊗ D + D ⊗ I), with D = tridiag(-1, 2, -1)
    B[mask_region_j, j] = 1 / num_nodes_in_region_j
"""

from typing import Union, Sequence
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .floorplan import N, REGION_KEYS, k, region_to_mask, t_d


def build_A() -> sp.csr_matrix:
    """Assemble the sparse 5-point Laplacian conductance matrix A.

    The continuous steady-state heat equation with Dirichlet boundary conditions
    is discretized using a central 5-point finite difference stencil:
        k * t_d * (4*θ_ij - θ_{i+1,j} - θ_{i-1,j} - θ_{i,j+1} - θ_{i,j-1}) = q_ij
    where the grid spacing h cancels out in 2D conduction through thickness t_d.

    Returns:
        sp.csr_matrix: Symmetric positive-definite conductance matrix of shape
        (N², N²) = (1600, 1600) with units of W/K.
    """
    # 1D tridiagonal second difference matrix: [-1, 2, -1]
    D = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(N, N), format="csr")
    I = sp.eye(N, format="csr")

    # Kronecker sum for 2D 5-point stencil: A = k * t_d * (I ⊗ D + D ⊗ I)
    kron_sum = sp.kron(I, D, format="csr") + sp.kron(D, I, format="csr")
    A = (k * t_d * kron_sum).tocsr()
    return A


def build_B() -> np.ndarray:
    """Construct the power allocation matrix B mapping block powers to nodes.

    For region j, B[mask_j, j] = 1 / num_nodes_j, so that multiplying B by a
    5-element block power vector q (W) yields the nodal power vector q_node (W).

    Returns:
        np.ndarray: Dense matrix of shape (N², 5) = (1600, 5) with units 1/node.
    """
    num_nodes = N * N
    num_blocks = len(REGION_KEYS)
    B = np.zeros((num_nodes, num_blocks), dtype=float)

    for j, key in enumerate(REGION_KEYS):
        mask_2d = region_to_mask(key)
        mask_flat = mask_2d.ravel()
        n_pts = int(np.sum(mask_flat))
        if n_pts > 0:
            B[mask_flat, j] = 1.0 / n_pts

    return B


def solve_temperature(q_map: np.ndarray, A: sp.spmatrix) -> np.ndarray:
    """Solve for the 2D temperature rise field θ given a nodal power map.

    Args:
        q_map: (N, N) array of per-node heat dissipation in Watts.
        A: (N², N²) sparse finite-difference conductance matrix in W/K.

    Returns:
        np.ndarray: (N, N) array of steady-state temperature rise θ = T - T_amb (K).
    """
    q_vec = q_map.ravel()
    theta_vec = spla.spsolve(A, q_vec)
    return theta_vec.reshape((N, N))


def solve_from_regions(
    q_vec: Union[np.ndarray, Sequence[float]],
    A: sp.spmatrix,
    B: np.ndarray,
) -> np.ndarray:
    """Solve for the 2D temperature rise field given a vector of 5 block powers.

    Args:
        q_vec: Array or sequence of length 5 containing total powers for
            [core0, core1, core2, core3, l2] in Watts.
        A: (N², N²) sparse finite-difference conductance matrix in W/K.
        B: (N², 5) power distribution matrix.

    Returns:
        np.ndarray: (N, N) array of steady-state temperature rise θ = T - T_amb (K).
    """
    q_arr = np.asarray(q_vec, dtype=float)
    q_nodes = B @ q_arr
    theta_vec = spla.spsolve(A, q_nodes)
    return theta_vec.reshape((N, N))
