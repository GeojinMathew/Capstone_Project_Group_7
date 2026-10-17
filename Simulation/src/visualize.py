"""Publication-quality visualization routines for 2D and 3D thermal fields.

This module provides plotting utilities using Matplotlib and mplot3d:
1. 2D temperature rise/field heatmaps with die floorplan annotations.
2. Grouped bar charts comparing uniform vs. LP-optimized block power allocations.
3. Power budget parameter sweep curves showing peak temperature scaling.
4. 3D surface plots of temperature fields with fixed Z scaling.
5. 3D extruded floorplan visualizations of regional temperature rises.
6. 3D rotating animated surface plots saved as GIF animations.
"""

from pathlib import Path
from typing import Optional, Sequence, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from .floorplan import N, REGIONS, T_amb, region_to_mask


def plot_heatmap(
    theta: np.ndarray,
    title: str,
    filename: Union[str, Path],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Generate and save a 2D temperature heatmap across the silicon die.

    Args:
        theta: (N, N) temperature rise array θ in Kelvin.
        title: Plot title.
        filename: Destination path for saving PNG figure.
        vmin: Optional minimum temperature for colorbar scaling (K).
        vmax: Optional maximum temperature for colorbar scaling (K).
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    temp_field = theta + T_amb

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    
    im = ax.imshow(
        temp_field,
        origin="lower",
        extent=[0.0, 10.0, 0.0, 10.0],
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (K)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    display_names = {
        "core0": "Core 0",
        "core1": "Core 1",
        "core2": "Core 2",
        "core3": "Core 3",
        "l2": "L2 Cache",
    }
    
    for key, (xmin, xmax, ymin, ymax) in REGIONS.items():
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1.2,
            edgecolor="cyan" if key != "l2" else "yellow",
            facecolor="none",
            linestyle="--",
            alpha=0.85,
        )
        ax.add_patch(rect)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        ax.text(
            cx,
            cy,
            display_names.get(key, key),
            color="white",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.45, edgecolor="none"),
        )

    peak_temp = np.max(temp_field)
    ax.set_title(f"{title}\n(Peak: {peak_temp:.2f} K)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("x (mm)", fontsize=11)
    ax.set_ylabel("y (mm)", fontsize=11)
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap: {filepath}")


def plot_power_bars(
    uniform_q: Union[np.ndarray, Sequence[float]],
    lp_q: Union[np.ndarray, Sequence[float]],
    filename: Union[str, Path],
) -> None:
    """Plot a grouped bar chart comparing uniform vs LP-optimal block powers.

    Args:
        uniform_q: Sequence of 5 block powers under uniform / baseline strategy (W).
        lp_q: Sequence of 5 block powers under LP-optimal strategy (W).
        filename: Destination path for saving PNG figure.
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Core 0", "Core 1", "Core 2", "Core 3", "L2 Cache"]
    x = np.arange(len(labels))
    width = 0.35

    q_u = np.asarray(uniform_q, dtype=float)
    q_opt = np.asarray(lp_q, dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=150)
    bars1 = ax.bar(x - width / 2, q_u, width, label="Uniform / Baseline", color="#4C72B0", edgecolor="black")
    bars2 = ax.bar(x + width / 2, q_opt, width, label="LP-Optimal", color="#DD8452", edgecolor="black")

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Power Allocation by Functional Block (P_max = 5.0 W)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Functional Block", fontsize=11)
    ax.set_ylabel("Power Allocation (W)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0.0, max(np.max(q_u), np.max(q_opt)) * 1.25)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved bar chart: {filepath}")


def plot_sweep(
    P_array: np.ndarray,
    peak_uniform: np.ndarray,
    peak_lp: np.ndarray,
    filename: Union[str, Path],
) -> None:
    """Plot peak die temperature as a function of total power budget P_max.

    Args:
        P_array: 1D array of power budget values in Watts.
        peak_uniform: 1D array of peak temperatures under uniform power (K).
        peak_lp: 1D array of peak temperatures under LP-optimal power (K).
        filename: Destination path for saving PNG figure.
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)

    ax.plot(
        P_array,
        peak_uniform,
        "o-",
        color="#C44E52",
        linewidth=2.2,
        markersize=6,
        label="Uniform Power Allocation",
    )
    ax.plot(
        P_array,
        peak_lp,
        "s--",
        color="#4C72B0",
        linewidth=2.2,
        markersize=6,
        label="LP-Optimal Allocation",
    )

    ax.fill_between(
        P_array,
        peak_lp,
        peak_uniform,
        color="#8172B3",
        alpha=0.22,
        label="Thermal Reduction ΔT",
    )

    idx_5w = np.argmin(np.abs(P_array - 5.0))
    p_5w = P_array[idx_5w]
    t_u_5w = peak_uniform[idx_5w]
    t_lp_5w = peak_lp[idx_5w]
    delta_5w = t_u_5w - t_lp_5w

    ax.annotate(
        f"ΔT = {delta_5w:.2f} K @ {p_5w:.1f} W",
        xy=(p_5w, 0.5 * (t_u_5w + t_lp_5w)),
        xytext=(p_5w - 1.8, 0.5 * (t_u_5w + t_lp_5w) + 3.0),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_title("Peak Die Temperature vs. Total Power Budget", fontsize=12, fontweight="bold")
    ax.set_xlabel("Total Power Budget P_max (W)", fontsize=11)
    ax.set_ylabel("Peak Temperature (K)", fontsize=11)
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"Saved sweep plot: {filepath}")


# =============================================================================
# 3D VISUALIZATIONS (mplot3d)
# =============================================================================

def _build_padded_grid(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad interior temperature rise θ with Dirichlet zero boundary conditions."""
    x_full = np.linspace(0.0, 10.0, N + 2)
    y_full = np.linspace(0.0, 10.0, N + 2)
    X_full, Y_full = np.meshgrid(x_full, y_full)
    Z_full = np.zeros((N + 2, N + 2), dtype=float)
    Z_full[1:-1, 1:-1] = theta
    T_full = T_amb + Z_full
    return X_full, Y_full, T_full


def plot_3d_surface(
    theta: np.ndarray,
    title: str,
    filename: Union[str, Path],
    zlim: tuple[float, float] = (300.0, 360.0),
) -> None:
    """Create a 3D surface plot of the temperature field across the silicon die.

    Args:
        theta: (N, N) temperature rise array θ in Kelvin.
        title: Plot title.
        filename: Destination path for saving PNG figure.
        zlim: Fixed range for Z-axis in Kelvin (default: (300.0, 360.0)).
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    X_full, Y_full, T_full = _build_padded_grid(theta)

    fig = plt.figure(figsize=(8.0, 6.2), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X_full,
        Y_full,
        T_full,
        cmap="inferno",
        antialiased=True,
        rstride=1,
        cstride=1,
        vmin=zlim[0],
        vmax=zlim[1],
        linewidth=0,
    )

    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    ax.set_zlim(zlim[0], zlim[1])

    ax.set_xlabel("x (mm)", fontsize=10, labelpad=8)
    ax.set_ylabel("y (mm)", fontsize=10, labelpad=8)
    ax.set_zlabel("Temperature (K)", fontsize=10, labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    ax.view_init(elev=30, azim=-60)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.1)
    cbar.set_label("Temperature (K)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_3d_floorplan(
    theta: np.ndarray,
    title: str,
    filename: Union[str, Path],
    zlim: tuple[float, float] = (300.0, 360.0),
) -> None:
    """Create a 3D extruded rectangular floorplan slab view of functional block temperatures.

    Each block is extruded from Z = 300 K to Z = 300 + (mean temperature rise in that region),
    colored by its mean temperature, with the die plate at Z = 300 K.

    Args:
        theta: (N, N) temperature rise array θ in Kelvin.
        title: Plot title.
        filename: Destination path for saving PNG figure.
        zlim: Fixed range for Z-axis in Kelvin (default: (300.0, 360.0)).
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.5, 6.5), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # Draw die base plate: 10mm x 10mm thin plate at Z = 300 K
    ax.bar3d(
        0.0,
        0.0,
        299.0,
        10.0,
        10.0,
        1.0,
        color="#E0E0E0",
        edgecolor="#B0B0B0",
        alpha=0.6,
        shade=True,
    )

    norm = colors.Normalize(vmin=zlim[0], vmax=zlim[1])
    cmap = cm.inferno

    display_names = {
        "core0": "core0",
        "core1": "core1",
        "core2": "core2",
        "core3": "core3",
        "l2": "L2",
    }

    for key, (xmin, xmax, ymin, ymax) in REGIONS.items():
        mask = region_to_mask(key)
        mean_rise = float(np.mean(theta[mask]))
        mean_t = T_amb + mean_rise

        dx = xmax - xmin
        dy = ymax - ymin
        dz = max(mean_rise, 0.1)  # small positive height for visibility if rise ~ 0

        slab_color = cmap(norm(mean_t))

        # Extrude slab from Z = 300 K to Z = 300 + mean_rise
        ax.bar3d(
            xmin,
            ymin,
            T_amb,
            dx,
            dy,
            dz,
            color=slab_color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.92,
            shade=True,
        )

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = T_amb + dz + 1.2
        ax.text(
            cx,
            cy,
            cz,
            f"{display_names.get(key, key)}\n({mean_t:.1f} K)",
            color="black",
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    ax.set_zlim(zlim[0], zlim[1])

    ax.set_xlabel("x (mm)", fontsize=10, labelpad=8)
    ax.set_ylabel("y (mm)", fontsize=10, labelpad=8)
    ax.set_zlabel("Temperature (K)", fontsize=10, labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    ax.view_init(elev=30, azim=-60)

    # Scalar mappable for colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, aspect=12, pad=0.1)
    cbar.set_label("Mean Temperature (K)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)


def plot_rotating_3d(
    theta: np.ndarray,
    filename: Union[str, Path],
    n_frames: int = 60,
    zlim: tuple[float, float] = (300.0, 360.0),
) -> None:
    """Create a 360-degree rotating animated 3D surface plot saved as a GIF.

    Args:
        theta: (N, N) temperature rise array θ in Kelvin.
        filename: Destination path for saving GIF animation.
        n_frames: Number of animation frames (default: 60).
        zlim: Fixed range for Z-axis in Kelvin (default: (300.0, 360.0)).
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    X_full, Y_full, T_full = _build_padded_grid(theta)

    fig = plt.figure(figsize=(7.5, 6.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X_full,
        Y_full,
        T_full,
        cmap="inferno",
        antialiased=True,
        rstride=1,
        cstride=1,
        vmin=zlim[0],
        vmax=zlim[1],
        linewidth=0,
    )

    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    ax.set_zlim(zlim[0], zlim[1])

    ax.set_xlabel("x (mm)", fontsize=9, labelpad=6)
    ax.set_ylabel("y (mm)", fontsize=9, labelpad=6)
    ax.set_zlabel("Temperature (K)", fontsize=9, labelpad=6)
    ax.set_title("3D Temperature Field (Rotating View)", fontsize=11, fontweight="bold", pad=8)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.08)
    cbar.set_label("Temperature (K)", fontsize=9, fontweight="bold")

    def update(frame: int):
        azim = -60.0 + frame * (360.0 / n_frames)
        ax.view_init(elev=30, azim=azim)
        return [surf]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(filepath, writer="pillow", fps=20)
    plt.close(fig)


# Alias function name for flexibility
plot_3d_rotating = plot_rotating_3d
