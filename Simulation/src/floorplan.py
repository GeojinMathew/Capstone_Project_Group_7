"""Floorplan and geometric discretization for the quad-core RISC-V die.

This module defines the physical parameters of the silicon die, the rectangular
regions for the functional blocks (4 RISC-V cores and shared L2 cache), and helper
routines to map physical coordinates to discrete 2D grid node masks.
"""

from typing import Dict, Tuple
import numpy as np

# Hardware and physical constants
L: float = 10e-3          # Die length and width: 10 mm (m)
N: int = 40               # Number of interior nodes per side
h: float = L / (N + 1)    # Grid spacing (m) ≈ 0.2439 mm
T_amb: float = 300.0      # Ambient heat sink temperature on all four edges (K)
P_max: float = 5.0        # Total die power budget (W)
t_d: float = 150e-6       # Silicon die thickness: 150 µm (m)
k: float = 150.0          # Thermal conductivity of silicon (W/(m·K))
rho_cp: float = 1.63e6    # Volumetric heat capacity of silicon (J/(m³·K))

# Rectangular functional block regions in millimetres: (x_min, x_max, y_min, y_max)
REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    "core0": (1.0, 4.5, 5.5, 9.0),  # top-left core
    "core1": (5.5, 9.0, 5.5, 9.0),  # top-right core
    "core2": (1.0, 4.5, 1.0, 4.5),  # bottom-left core
    "core3": (5.5, 9.0, 1.0, 4.5),  # bottom-right core
    "l2": (3.5, 6.5, 4.0, 6.0),     # central shared L2 cache
}

REGION_KEYS = ["core0", "core1", "core2", "core3", "l2"]


def node_coordinates() -> Tuple[np.ndarray, np.ndarray]:
    """Compute physical 2D coordinates (X, Y) for interior grid nodes in metres.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 2D coordinate matrices (X, Y) in metres,
        each having shape (N, N).
    """
    x = np.arange(1, N + 1) * h
    y = np.arange(1, N + 1) * h
    X, Y = np.meshgrid(x, y)
    return X, Y


def region_to_mask(region_name: str) -> np.ndarray:
    """Generate a boolean mask for nodes falling within a given functional block.

    Args:
        region_name: Identifier for the block ('core0', 'core1', 'core2', 'core3', 'l2',
            or case-insensitive variations like 'L2 cache').

    Returns:
        np.ndarray: (N, N) boolean array where True indicates the node lies within
        the specified rectangular region.

    Raises:
        KeyError: If the region_name cannot be resolved.
    """
    key = region_name.strip().lower()
    if key in ("l2_cache", "l2 cache", "l2cache"):
        key = "l2"

    if key not in REGIONS:
        raise KeyError(
            f"Unknown region '{region_name}'. Available regions: {list(REGIONS.keys())}"
        )

    xmin_mm, xmax_mm, ymin_mm, ymax_mm = REGIONS[key]
    X, Y = node_coordinates()
    X_mm = X * 1e3
    Y_mm = Y * 1e3

    mask = (
        (X_mm >= xmin_mm)
        & (X_mm <= xmax_mm)
        & (Y_mm >= ymin_mm)
        & (Y_mm <= ymax_mm)
    )
    return mask
