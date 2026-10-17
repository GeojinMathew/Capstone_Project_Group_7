"""Power dissipation models for the quad-core RISC-V processor.

This module provides power models representing active and idle core states
along with the shared L2 cache, mapping block-level power budgets to spatially
resolved per-node dissipation maps (W per node).
"""

from typing import Dict, Optional, Sequence, Tuple
import numpy as np

from .floorplan import N, REGION_KEYS, region_to_mask

# Synthetic RISC-V power constants based on realistic hardware data (Watts)
P_ACTIVE: float = 1.2  # Power consumption of an active core (W)
P_IDLE: float = 0.3    # Power consumption of an idle core (W)
P_L2: float = 0.6      # Power consumption of the shared L2 cache (W)


def block_powers(active_cores: Sequence[int] = (0, 1)) -> Dict[str, float]:
    """Calculate total power dissipation for each functional block.

    Args:
        active_cores: Sequence of integer indices (0 to 3) indicating which cores
            are in the active state. Cores not in this sequence are set to idle.

    Returns:
        Dict[str, float]: Mapping of block name ('core0', 'core1', 'core2', 'core3', 'l2')
        to its power consumption in Watts.
    """
    powers: Dict[str, float] = {}
    active_set = set(active_cores)

    for i in range(4):
        powers[f"core{i}"] = P_ACTIVE if i in active_set else P_IDLE

    powers["l2"] = P_L2
    return powers


def build_power_map(
    active_cores: Sequence[int] = (0, 1),
    total_power: Optional[float] = None,
) -> np.ndarray:
    """Generate a spatially-resolved 2D power map across the die grid (Watts per node).

    Power is distributed uniformly across all interior nodes enclosed by each
    functional block region. If a node is shared by multiple overlapping regions,
    dissipated powers are summed.

    Args:
        active_cores: Sequence of core indices operating at active power level.
        total_power: Optional target total power (W). If provided, the resulting
            power map is linearly scaled so that np.sum(power_map) == total_power.

    Returns:
        np.ndarray: (N, N) array containing power dissipation in Watts at each
        interior grid node.
    """
    powers = block_powers(active_cores=active_cores)
    q_map = np.zeros((N, N), dtype=float)

    for key in REGION_KEYS:
        mask = region_to_mask(key)
        num_nodes = int(np.sum(mask))
        if num_nodes > 0:
            power_per_node = powers[key] / num_nodes
            q_map[mask] += power_per_node

    if total_power is not None:
        current_total = float(np.sum(q_map))
        if current_total > 0.0:
            q_map *= total_power / current_total

    return q_map
