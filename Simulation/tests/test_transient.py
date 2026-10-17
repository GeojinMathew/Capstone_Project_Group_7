"""Verification and performance tests for the transient thermal solver."""

import time
import numpy as np
import pytest
import scipy.sparse.linalg as spla

from src.floorplan import N
from src.thermal_solver import build_A, build_B
from src.transient import transient_solve


def test_transient_convergence_to_steady():
    """Assert transient solution converges to steady-state solution within 1e-6 relative error at n_steps=200."""
    A = build_A()
    B = build_B()
    powers = [1.25, 1.25, 1.25, 1.25, 0.0]

    # Run transient for 200 steps (dt = 5 ms -> 1.0 s total simulated time)
    fields = transient_solve(block_powers=powers, dt=5e-3, n_steps=200, A=A, B=B)
    theta_transient_final = fields[-1]

    # Direct steady state solve
    q_nodes = B @ np.array(powers, dtype=float)
    theta_steady = spla.spsolve(A, q_nodes).reshape((N, N))

    rel_err = np.linalg.norm(theta_transient_final - theta_steady) / np.linalg.norm(theta_steady)
    assert rel_err < 1e-6, f"Transient solution did not converge to steady state: rel_err={rel_err:.3e} >= 1e-6"


def test_transient_monotonic_temperature():
    """Assert maximum temperature is monotonic non-decreasing over time starting from θ=0."""
    A = build_A()
    B = build_B()
    powers = [1.2, 1.2, 0.3, 0.3, 0.6]

    fields = transient_solve(block_powers=powers, dt=5e-3, n_steps=100, A=A, B=B)
    peak_trajectory = [float(np.max(f)) for f in fields]

    # Differences should all be non-negative (within rounding tolerance)
    diffs = np.diff(peak_trajectory)
    assert np.all(diffs >= -1e-12), "Maximum temperature decreased during heating from zero state"


def test_transient_execution_speed():
    """Assert total runtime of transient_solve(100 steps) is under 3.0 seconds."""
    powers = [1.25, 1.25, 1.25, 1.25, 0.0]
    t0 = time.time()
    fields = transient_solve(block_powers=powers, dt=5e-3, n_steps=100)
    elapsed = time.time() - t0

    assert len(fields) == 101
    assert elapsed < 3.0, f"Transient solve too slow: took {elapsed:.2f} s >= 3.0 s"
