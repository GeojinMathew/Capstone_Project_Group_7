"""Transient thermal time-stepping simulation using Backward Euler.

This module models the time-dependent heat spreading across the silicon die:
    C * dθ/dt + A * θ = B * q
where C = ρ * cp * t_d * h² * I is the diagonal thermal mass matrix (J/K),
and A is the sparse 5-point discrete Laplacian conductance matrix (W/K).

Backward Euler time discretization:
    (C / dt + A) * θ^{n+1} = (C / dt) * θ^n + B * q
Pre-factorizing (C / dt + A) allows step evaluation in sub-millisecond time.
"""

from typing import Callable, List, Optional, Sequence, Union
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .floorplan import N, h, rho_cp, t_d
from .thermal_solver import build_A, build_B


def transient_solve(
    block_powers: Union[np.ndarray, Sequence[float]],
    dt: float = 5e-3,
    n_steps: int = 100,
    A: Optional[sp.spmatrix] = None,
    B: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Simulate transient temperature rise evolution via Backward Euler time-stepping.

    Args:
        block_powers: 5-element sequence of total powers [core0..core3, L2] in Watts.
        dt: Time step size in seconds (default: 5e-3 s = 5 ms).
        n_steps: Number of integration steps (default: 100, yielding 0.5 s total).
        A: Optional pre-assembled conductance matrix (N², N²). If None, builds A.
        B: Optional pre-assembled power distribution matrix (N², 5). If None, builds B.

    Returns:
        List[np.ndarray]: List of (n_steps + 1) temperature rise fields θ,
        each formatted as an (N, N) numpy array in Kelvin, starting from θ⁰ = 0.
    """
    if A is None:
        A = build_A()
    if B is None:
        B = build_B()

    q_vec = np.asarray(block_powers, dtype=float)
    rhs_power = B @ q_vec

    # Diagonal thermal capacitance mass matrix C = ρ * cp * t_d * h² * I
    c_cell = float(rho_cp * t_d * (h**2))
    C = sp.diags([c_cell] * (N * N), 0, format="csr")

    # System matrix M = (C / dt + A)
    c_div_dt = C / dt
    M = c_div_dt + A

    # Pre-factorize M once for fast repeated triangular solves
    solve_step = spla.factorized(M.tocsc())

    theta = np.zeros(N * N, dtype=float)
    fields = [theta.reshape((N, N)).copy()]

    for _ in range(n_steps):
        rhs = (c_div_dt @ theta) + rhs_power
        theta = solve_step(rhs)
        fields.append(theta.reshape((N, N)).copy())

    return fields


def animate_in_gui(
    canvas,
    fields: Sequence[np.ndarray],
    root,
    update_callback: Callable[[np.ndarray, int, int], None],
    framerate: int = 30,
    on_complete: Optional[Callable[[], None]] = None,
) -> None:
    """Non-blocking animated playback of transient fields inside a Tkinter GUI.

    Iterates through simulation snapshots using root.after() delays and canvas.draw_idle(),
    preventing GUI lockup and allowing concurrent user interaction.

    Args:
        canvas: matplotlib FigureCanvasTkAgg instance.
        fields: Sequence of (N, N) temperature fields to render.
        root: Tkinter root window or widget with .after() scheduling method.
        update_callback: Function called on each frame: update_callback(field, frame_idx, total_frames).
        framerate: Playback target frames per second (default: 30).
        on_complete: Optional callback executed when animation reaches final frame.
    """
    delay_ms = max(1, int(1000.0 / framerate))
    total_frames = len(fields)

    def _step(frame_idx: int) -> None:
        if frame_idx >= total_frames:
            if on_complete is not None:
                on_complete()
            return

        current_field = fields[frame_idx]
        update_callback(current_field, frame_idx, total_frames)
        canvas.draw_idle()

        # Schedule next frame non-blockingly
        root.after(delay_ms, lambda: _step(frame_idx + 1))

    # Kick off first step
    _step(0)
