"""Live Tkinter Dashboard and Interactive Transient Simulator.

An interactive multi-panel GUI for real-time thermal simulation and power budgeting
of a quad-core RISC-V processor on a silicon die.

Features:
- Three View Modes: 3D Surface (with block labels), 2D Heatmap, and 3D Labeled Floorplan
- Real-time live updating with fixed temperature scaling (300 K to 360 K)
- Interactive power budget slider (1.0 W to 8.0 W) with instantaneous LP re-solve
- Scenario toggle (Symmetric vs. Asymmetric) and Animation Mode selection
- Backward Euler transient heat spreading animation (0.5 s in ~30 fps playback)
- Embedded per-block power allocation bar chart
- Live metric readouts, thermal diagnostics, and block labels toggle
- Standalone --screenshot CLI flag rendering a 3-panel comparison figure
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as patches
from matplotlib import cm, colors
import numpy as np

# Add local directory to path for standalone launch
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.floorplan import N, P_max as DEFAULT_P_MAX, REGIONS, T_amb, region_to_mask
from src.optimizer import solve_lp
from src.power_model import block_powers
from src.thermal_solver import build_A, build_B, solve_from_regions
from src.transient import transient_solve
from src.visualize import _build_padded_grid

BLOCK_NAMES = ["core0", "core1", "core2", "core3", "l2"]
DISPLAY_NAMES = {
    "core0": "Core 0",
    "core1": "Core 1",
    "core2": "Core 2",
    "core3": "Core 3",
    "l2": "L2",
}


# =============================================================================
# REUSABLE VIEW BUILDER ROUTINES (Used by GUI, Screenshot, and Unit Tests)
# =============================================================================

def build_3d_surface(
    ax: matplotlib.axes.Axes,
    theta: np.ndarray,
    title_suffix: str = "",
    show_labels: bool = True,
) -> Tuple[object, List[matplotlib.text.Text]]:
    """Build a 3D temperature surface plot on the provided 3D axes with centroid labels."""
    X_f, Y_f, T_f = _build_padded_grid(theta)
    surf = ax.plot_surface(
        X_f,
        Y_f,
        T_f,
        cmap="inferno",
        vmin=300.0,
        vmax=360.0,
        antialiased=True,
        linewidth=0,
    )
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    ax.set_zlim(300.0, 360.0)
    ax.set_xlabel("x (mm)", fontsize=9, labelpad=5)
    ax.set_ylabel("y (mm)", fontsize=9, labelpad=5)
    ax.set_zlabel("T (K)", fontsize=9, labelpad=5)
    ax.view_init(elev=30, azim=-60)

    label_texts = []
    if show_labels:
        for key in BLOCK_NAMES:
            xmin, xmax, ymin, ymax = REGIONS[key]
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            mask = region_to_mask(key)
            mean_theta = float(np.mean(theta[mask]))
            z = 300.0 + mean_theta + 2.0
            label = DISPLAY_NAMES[key]
            t = ax.text(
                cx,
                cy,
                z,
                label,
                fontsize=9,
                color="white",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
            )
            label_texts.append(t)

    peak_temp = float(T_amb + np.max(theta))
    suffix = f" {title_suffix}" if title_suffix else ""
    ax.set_title(f"3D Surface{suffix} — Peak {peak_temp:.2f} K", fontsize=11, fontweight="bold")
    return surf, label_texts


def build_2d_heatmap(
    ax: matplotlib.axes.Axes,
    theta: np.ndarray,
    title_suffix: str = "",
    show_labels: bool = True,
) -> Tuple[matplotlib.image.AxesImage, List[patches.Rectangle], List[matplotlib.text.Text]]:
    """Build a 2D temperature heatmap with cyan functional block outlines and centroid labels."""
    _, _, T_f = _build_padded_grid(theta)
    im = ax.imshow(
        T_f,
        origin="lower",
        extent=[0.0, 10.0, 0.0, 10.0],
        cmap="inferno",
        vmin=300.0,
        vmax=360.0,
        interpolation="bilinear",
    )

    rect_patches = []
    for key in BLOCK_NAMES:
        xmin, xmax, ymin, ymax = REGIONS[key]
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2.0,
            edgecolor="cyan",
            facecolor="none",
        )
        ax.add_patch(rect)
        rect_patches.append(rect)

    label_texts = []
    for key in BLOCK_NAMES:
        xmin, xmax, ymin, ymax = REGIONS[key]
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        t = ax.text(
            cx,
            cy,
            DISPLAY_NAMES[key],
            fontsize=9,
            color="white",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
            visible=show_labels,
        )
        label_texts.append(t)

    peak_temp = float(T_amb + np.max(theta))
    suffix = f" {title_suffix}" if title_suffix else ""
    ax.set_title(f"2D Heatmap{suffix} — Peak {peak_temp:.2f} K", fontsize=11, fontweight="bold")
    ax.set_xlabel("x (mm)", fontsize=9)
    ax.set_ylabel("y (mm)", fontsize=9)
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    return im, rect_patches, label_texts


def build_3d_floorplan(
    ax: matplotlib.axes.Axes,
    theta: np.ndarray,
    title_suffix: str = "",
    show_labels: bool = True,
) -> List[matplotlib.text.Text]:
    """Build extruded 3D rectangular block slabs with top-face labels on a flat die plate."""
    ax.cla()

    # Thin flat die plate at Z = 300 K (light grey, alpha 0.3)
    ax.bar3d(
        0.0,
        0.0,
        299.5,
        10.0,
        10.0,
        0.5,
        color="#D3D3D3",
        edgecolor="#C0C0C0",
        alpha=0.3,
        shade=True,
    )

    norm = colors.Normalize(vmin=300.0, vmax=360.0)
    cmap = cm.inferno

    label_texts = []
    for key in BLOCK_NAMES:
        xmin, xmax, ymin, ymax = REGIONS[key]
        mask = region_to_mask(key)
        mean_rise = float(np.mean(theta[mask]))
        mean_t = 300.0 + mean_rise
        local_peak = 300.0 + float(np.max(theta[mask]))

        dx = xmax - xmin
        dy = ymax - ymin
        dz = max(mean_rise, 0.1)

        slab_color = cmap(norm(mean_t))

        ax.bar3d(
            xmin,
            ymin,
            300.0,
            dx,
            dy,
            dz,
            color=slab_color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
            shade=True,
        )

        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 300.0 + dz + 1.2
        label_text = f"{DISPLAY_NAMES[key]}\n{local_peak:.1f} K"
        t = ax.text(
            cx,
            cy,
            cz,
            label_text,
            color="white",
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6, edgecolor="none"),
            visible=show_labels,
        )
        label_texts.append(t)

    peak_temp = float(T_amb + np.max(theta))
    suffix = f" {title_suffix}" if title_suffix else ""
    ax.set_title(f"3D Floorplan{suffix} — Peak {peak_temp:.2f} K", fontsize=11, fontweight="bold")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 10.0)
    ax.set_zlim(300.0, 360.0)
    ax.set_xlabel("x (mm)", fontsize=9, labelpad=5)
    ax.set_ylabel("y (mm)", fontsize=9, labelpad=5)
    ax.set_zlabel("T (K)", fontsize=9, labelpad=5)
    ax.view_init(elev=30, azim=-60)
    return label_texts


def create_2d_heatmap_figure(
    theta: np.ndarray,
    title_suffix: str = "",
    show_labels: bool = True,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.image.AxesImage]:
    """Helper to create a standalone 2D heatmap figure for testing and analysis."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=100)
    im, _, _ = build_2d_heatmap(ax, theta, title_suffix=title_suffix, show_labels=show_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (K)", fontsize=10, fontweight="bold")
    return fig, ax, im


def create_3d_surface_figure(
    theta: np.ndarray,
    title_suffix: str = "",
    show_labels: bool = True,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Helper to create a standalone 3D surface figure for testing."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7.0, 5.5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    surf, _ = build_3d_surface(ax, theta, title_suffix=title_suffix, show_labels=show_labels)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.08)
    cbar.set_label("Temperature (K)", fontsize=9, fontweight="bold")
    return fig, ax


# =============================================================================
# HEADLESS 3-PANEL SCREENSHOT RENDERER
# =============================================================================

def render_headless_screenshot(output_path: Path) -> None:
    """Render a 3-panel screenshot (3D Surface, 2D Heatmap, 3D Labeled Floorplan) side-by-side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    print("[GUI] Rendering headless dashboard screenshot with Matplotlib Agg...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    A = build_A()
    B = build_B()
    p_budget = 5.0

    # Default scenario: Asymmetric (active_cores=(0, 1)), P_max=5.0
    lp_res = solve_lp(A, B, P_max=p_budget)
    theta_lp = lp_res["theta"]
    peak_lp = float(T_amb + lp_res["t_peak"])

    fig = plt.figure(figsize=(18.0, 6.0), dpi=120, facecolor="#F8F9FA")

    # Panel 1 (left): 3D Surface
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    surf, _ = build_3d_surface(ax1, theta_lp, title_suffix="(LP-Optimal)", show_labels=True)
    cbar1 = fig.colorbar(surf, ax=ax1, shrink=0.55, aspect=12, pad=0.08)
    cbar1.set_label("Temperature (K)", fontsize=9, fontweight="bold")

    # Panel 2 (middle): 2D Heatmap
    ax2 = fig.add_subplot(1, 3, 2)
    im, _, _ = build_2d_heatmap(ax2, theta_lp, title_suffix="(LP-Optimal)", show_labels=True)
    cbar2 = fig.colorbar(im, ax=ax2, shrink=0.75, pad=0.04)
    cbar2.set_label("Temperature (K)", fontsize=9, fontweight="bold")

    # Panel 3 (right): 3D Labeled Floorplan
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    build_3d_floorplan(ax3, theta_lp, title_suffix="(LP-Optimal)", show_labels=True)
    norm = colors.Normalize(vmin=300.0, vmax=360.0)
    sm = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
    sm.set_array([])
    cbar3 = fig.colorbar(sm, ax=ax3, shrink=0.55, aspect=12, pad=0.08)
    cbar3.set_label("Temperature (K)", fontsize=9, fontweight="bold")

    fig.suptitle(
        f"RISC-V Thermal-Aware Power Budgeting — Multi-View Thermal Profile (Asymmetric, P_max = {p_budget:.1f} W, Peak = {peak_lp:.2f} K)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.10, wspace=0.25)

    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"[GUI] Saved 3-panel screenshot to {output_path}")


# =============================================================================
# INTERACTIVE TKINTER APPLICATION
# =============================================================================

def main() -> None:
    """Launch interactive Tkinter GUI or render screenshot if --screenshot flag passed."""
    parser = argparse.ArgumentParser(description="Live Tkinter Thermal Simulator")
    parser.add_argument("--screenshot", action="store_true", help="Render screenshot to figures/gui_screenshot.png and exit")
    args = parser.parse_args()

    figures_dir = current_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.screenshot:
        screenshot_path = figures_dir / "gui_screenshot.png"
        render_headless_screenshot(screenshot_path)
        return

    # Standard Tkinter interactive launch
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt

    print("[GUI] Launching live thermal simulator")

    # Load system matrices once
    A = build_A()
    B = build_B()
    print(f"[GUI] Loaded A ({A.shape[0]}x{A.shape[1]}), B ({B.shape[0]}x{B.shape[1]})")

    root = tk.Tk()
    root.title("RISC-V Thermal-Aware Power Budgeting — Live Simulator")
    root.geometry("1400x800")
    root.minsize(1100, 650)

    # -------------------------------------------------------------------------
    # State variables
    # -------------------------------------------------------------------------
    p_budget_var = tk.DoubleVar(value=DEFAULT_P_MAX)
    scenario_var = tk.StringVar(value="Symmetric")              # "Symmetric" or "Asymmetric"
    view_type_var = tk.StringVar(value="3D Surface")            # "3D Surface", "2D Heatmap", "3D Labeled Floorplan"
    animation_var = tk.StringVar(value="Steady State (Fixed)")   # "Steady State (Fixed)", "Transient Playback"
    show_labels_var = tk.BooleanVar(value=True)                 # Show/hide block labels
    status_var = tk.StringVar(value="Optimal")

    # Current solved data
    current_state = {
        "theta": np.zeros((N, N)),
        "q": np.array([1.175, 1.175, 1.175, 1.175, 0.30]),
        "uniform_q": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "peak_uniform": 325.18,
        "peak_lp": 322.86,
        "reduction": 2.31,
        "lp_time": 0.005,
        "is_lp": True,
        "is_animating": False,
        "sim_time": "Steady",
        "title_suffix": "(LP-Optimal)",
    }

    # Internal references for live canvas updating
    view_refs = {
        "im": None,             # 2D heatmap AxesImage
        "surf": None,           # 3D surface Poly3DCollection
        "labels": [],           # List of Text artists
        "cbar": None,           # Active colorbar
        "ax": None,             # Active Axes
    }

    # -------------------------------------------------------------------------
    # Layout Frames (Three Columns)
    # -------------------------------------------------------------------------
    root.columnconfigure(0, weight=55)
    root.columnconfigure(1, weight=25)
    root.columnconfigure(2, weight=20)
    root.rowconfigure(0, weight=1)

    left_frame = ttk.Frame(root, padding=6)
    left_frame.grid(row=0, column=0, sticky="nsew")

    middle_frame = ttk.LabelFrame(root, text=" Control Panel ", padding=12)
    middle_frame.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

    right_frame = ttk.LabelFrame(root, text=" Live Results & Diagnostics ", padding=12)
    right_frame.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)

    # -------------------------------------------------------------------------
    # Left Column: Matplotlib View Canvas & Power Bar Chart
    # -------------------------------------------------------------------------
    left_frame.rowconfigure(0, weight=7)
    left_frame.rowconfigure(1, weight=3)
    left_frame.columnconfigure(0, weight=1)

    # Top figure (View modes: 3D Surface / 2D Heatmap / 3D Floorplan)
    fig_top = plt.Figure(figsize=(7.0, 4.8), dpi=100)
    canvas_top = FigureCanvasTkAgg(fig_top, master=left_frame)
    canvas_top.get_tk_widget().grid(row=0, column=0, sticky="nsew", pady=(0, 4))

    # Power Bar Chart Figure
    fig_bar = plt.Figure(figsize=(7.0, 2.2), dpi=100)
    ax_bar = fig_bar.add_subplot(111)
    bar_labels = ["Core 0", "Core 1", "Core 2", "Core 3", "L2"]
    x_indices = np.arange(len(bar_labels))
    bar_width = 0.38
    bars_uniform = ax_bar.bar(x_indices - bar_width/2, [1.0]*5, bar_width, label="Uniform", color="#4C72B0", edgecolor="black")
    bars_lp = ax_bar.bar(x_indices + bar_width/2, [1.0]*5, bar_width, label="LP-Optimal", color="#DD8452", edgecolor="black")
    ax_bar.set_xticks(x_indices)
    ax_bar.set_xticklabels(bar_labels, fontsize=9)
    ax_bar.set_ylabel("Power (W)", fontsize=9)
    ax_bar.set_ylim(0, 2.5)
    ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.grid(axis="y", linestyle=":", alpha=0.6)
    fig_bar.tight_layout()

    canvas_bar = FigureCanvasTkAgg(fig_bar, master=left_frame)
    canvas_bar.get_tk_widget().grid(row=1, column=0, sticky="nsew")

    # -------------------------------------------------------------------------
    # Right Column: Live Results Text Widget & Legend
    # -------------------------------------------------------------------------
    right_frame.rowconfigure(0, weight=6)
    right_frame.rowconfigure(1, weight=4)
    right_frame.columnconfigure(0, weight=1)

    results_text = tk.Text(right_frame, width=30, height=14, font=("Consolas", 10), wrap="word", relief="groove")
    results_text.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

    legend_text = tk.Text(right_frame, width=30, height=10, font=("Segoe UI", 9), wrap="word", relief="flat", bg="#F4F6F8")
    legend_text.grid(row=1, column=0, sticky="nsew")

    legend_content = (
        "COLOR & REGION LEGEND:\n"
        "------------------------------\n"
        "• Blue / Dark Purple:\n"
        "  Cool perimeter heat sink (300 K)\n"
        "• Orange / Yellow:\n"
        "  Hot active silicon core regions\n"
        "• Central Rectangular Block:\n"
        "  Shared L2 cache (long heat path)\n"
        "• Outer Die Boundary:\n"
        "  Fixed Dirichlet boundary (heat sink)\n\n"
        "VIEW MODES:\n"
        "• 3D Surface: Smooth spatial field\n"
        "• 2D Heatmap: Planar die floorplan\n"
        "• 3D Floorplan: Extruded block slabs"
    )
    legend_text.insert("1.0", legend_content)
    legend_text.config(state="disabled")

    # -------------------------------------------------------------------------
    # Canvas Rebuilding & Live Updating Functions
    # -------------------------------------------------------------------------
    def rebuild_view_canvas() -> None:
        """Fully reinitialize the top matplotlib subplot for the selected View Type."""
        fig_top.clear()
        view_type = view_type_var.get()
        theta_curr = current_state["theta"]
        show_lbl = show_labels_var.get()
        title_suffix = current_state["title_suffix"]

        if view_type == "2D Heatmap":
            ax = fig_top.add_subplot(111)
            im, _, labels = build_2d_heatmap(ax, theta_curr, title_suffix=title_suffix, show_labels=show_lbl)
            cbar = fig_top.colorbar(im, ax=ax, shrink=0.75, pad=0.04)
            cbar.set_label("Temperature (K)", fontsize=9, fontweight="bold")
            view_refs["im"] = im
            view_refs["surf"] = None
            view_refs["labels"] = labels
            view_refs["ax"] = ax
            view_refs["cbar"] = cbar

        elif view_type == "3D Labeled Floorplan":
            ax = fig_top.add_subplot(111, projection="3d")
            labels = build_3d_floorplan(ax, theta_curr, title_suffix=title_suffix, show_labels=show_lbl)
            norm = colors.Normalize(vmin=300.0, vmax=360.0)
            sm = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
            sm.set_array([])
            cbar = fig_top.colorbar(sm, ax=ax, shrink=0.55, aspect=12, pad=0.08)
            cbar.set_label("Temperature (K)", fontsize=9, fontweight="bold")
            view_refs["im"] = None
            view_refs["surf"] = None
            view_refs["labels"] = labels
            view_refs["ax"] = ax
            view_refs["cbar"] = cbar

        else:  # "3D Surface"
            ax = fig_top.add_subplot(111, projection="3d")
            surf, labels = build_3d_surface(ax, theta_curr, title_suffix=title_suffix, show_labels=show_lbl)
            cbar = fig_top.colorbar(surf, ax=ax, shrink=0.55, aspect=12, pad=0.08)
            cbar.set_label("Temperature (K)", fontsize=9, fontweight="bold")
            view_refs["im"] = None
            view_refs["surf"] = surf
            view_refs["labels"] = labels
            view_refs["ax"] = ax
            view_refs["cbar"] = cbar

        canvas_top.draw_idle()

    def update_displays(theta_field: np.ndarray, title_suffix: str = "") -> None:
        """Live-update current top view and bar chart without full reconstruction when possible."""
        current_state["theta"] = theta_field
        current_state["title_suffix"] = title_suffix
        view_type = view_type_var.get()
        show_lbl = show_labels_var.get()

        # 1. Update Top Visualization
        if view_type == "2D Heatmap" and view_refs["im"] is not None:
            # Efficient live update: set_data and update title without rebuilding figure
            _, _, T_f = _build_padded_grid(theta_field)
            view_refs["im"].set_data(T_f)
            peak_temp = float(T_amb + np.max(theta_field))
            suffix = f" {title_suffix}" if title_suffix else ""
            view_refs["ax"].set_title(f"2D Heatmap{suffix} — Peak {peak_temp:.2f} K", fontsize=11, fontweight="bold")
            canvas_top.draw_idle()

        elif view_type == "3D Surface" and view_refs["ax"] is not None and view_refs["surf"] is not None:
            # Update 3D surface artist and labels
            ax = view_refs["ax"]
            try:
                view_refs["surf"].remove()
            except Exception:
                pass

            for t in view_refs["labels"]:
                try:
                    t.remove()
                except Exception:
                    pass
            view_refs["labels"].clear()

            X_f, Y_f, T_f = _build_padded_grid(theta_field)
            new_surf = ax.plot_surface(
                X_f,
                Y_f,
                T_f,
                cmap="inferno",
                vmin=300.0,
                vmax=360.0,
                antialiased=True,
                linewidth=0,
            )
            view_refs["surf"] = new_surf

            if show_lbl:
                for key in BLOCK_NAMES:
                    xmin, xmax, ymin, ymax = REGIONS[key]
                    cx = 0.5 * (xmin + xmax)
                    cy = 0.5 * (ymin + ymax)
                    mask = region_to_mask(key)
                    mean_theta = float(np.mean(theta_field[mask]))
                    z = 300.0 + mean_theta + 2.0
                    label = DISPLAY_NAMES[key]
                    t = ax.text(
                        cx,
                        cy,
                        z,
                        label,
                        fontsize=9,
                        color="white",
                        ha="center",
                        va="center",
                        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
                    )
                    view_refs["labels"].append(t)

            peak_temp = float(T_amb + np.max(theta_field))
            suffix = f" {title_suffix}" if title_suffix else ""
            ax.set_title(f"3D Surface{suffix} — Peak {peak_temp:.2f} K", fontsize=11, fontweight="bold")
            canvas_top.draw_idle()

        elif view_type == "3D Labeled Floorplan" and view_refs["ax"] is not None:
            # Re-render extruded floorplan
            ax = view_refs["ax"]
            labels = build_3d_floorplan(ax, theta_field, title_suffix=title_suffix, show_labels=show_lbl)
            view_refs["labels"] = labels
            canvas_top.draw_idle()

        else:
            rebuild_view_canvas()

        # 2. Update Bar Chart
        q_unif = current_state["uniform_q"]
        q_lp = current_state["q_lp"]
        is_lp_active = current_state["is_lp"]

        for rect, val in zip(bars_uniform, q_unif):
            rect.set_height(val)
            rect.set_alpha(0.4 if is_lp_active else 1.0)
            rect.set_linewidth(2.0 if not is_lp_active else 0.8)
        for rect, val in zip(bars_lp, q_lp):
            rect.set_height(val)
            rect.set_alpha(1.0 if is_lp_active else 0.4)
            rect.set_linewidth(2.0 if is_lp_active else 0.8)

        active_tag = "LP-OPTIMAL" if is_lp_active else "UNIFORM BASELINE"
        p_val_num = float(p_budget_var.get())
        ax_bar.set_title(f"Power Allocation — Displaying: {active_tag} (Total: {p_val_num:.2f} W)", fontsize=10, fontweight="bold")

        max_val = max(float(np.max(q_unif)), float(np.max(q_lp)), 1.5)
        ax_bar.set_ylim(0, max_val * 1.25)
        canvas_bar.draw_idle()

        # 3. Update Results Table Text
        results_text.config(state="normal")
        results_text.delete("1.0", "end")
        summary_str = (
            f"=== LIVE METRICS ===\n\n"
            f"DISPLAYING:      {active_tag}\n"
            f"Peak (uniform):  {current_state['peak_uniform']:>8.2f} K\n"
            f"Peak (LP):       {current_state['peak_lp']:>8.2f} K\n"
            f"Reduction (ΔT):  {current_state['reduction']:>8.2f} K\n"
            f"Active powers:   {float(np.sum(current_state['q'])):>8.4f} W\n"
            f"LP solve time:   {current_state['lp_time']:>8.4f} s\n"
            f"Simulated time:  {str(current_state['sim_time']):>8}\n\n"
            f"STATUS:\n"
            f"  {status_var.get()}"
        )
        results_text.insert("1.0", summary_str)
        results_text.config(state="disabled")

    def solve_current() -> None:
        """Compute steady-state solutions (both uniform and LP) for current settings."""
        p_val = float(p_budget_var.get())
        is_symmetric = (scenario_var.get() == "Symmetric")

        status_var.set("Solving LP...")
        root.update_idletasks()

        cores = (0, 1, 2, 3) if is_symmetric else (0, 1)
        raw_p = block_powers(active_cores=cores)
        raw_q = np.array([raw_p["core0"], raw_p["core1"], raw_p["core2"], raw_p["core3"], raw_p["l2"]])
        q_uniform = raw_q * (p_val / np.sum(raw_q))

        theta_uniform = solve_from_regions(q_uniform, A, B)
        peak_uniform = float(T_amb + np.max(theta_uniform))

        t0 = time.time()
        lp_res = solve_lp(A, B, P_max=p_val)
        lp_time = time.time() - t0

        q_lp = lp_res["q"]
        theta_lp = lp_res["theta"]
        peak_lp = float(T_amb + lp_res["t_peak"])

        current_state["uniform_q"] = q_uniform
        current_state["theta_uniform"] = theta_uniform
        current_state["theta_lp"] = theta_lp
        current_state["q_lp"] = q_lp
        current_state["peak_uniform"] = peak_uniform
        current_state["peak_lp"] = peak_lp
        current_state["reduction"] = peak_uniform - peak_lp
        current_state["lp_time"] = lp_time
        current_state["sim_time"] = "Steady"

        if current_state["is_lp"]:
            status_var.set("Optimal (LP Active)" if lp_res["success"] else "Infeasible")
            current_state["theta"] = theta_lp
            current_state["q"] = q_lp
            update_displays(theta_lp, title_suffix="(LP-Optimal)")
        else:
            status_var.set("Uniform Baseline Active")
            current_state["theta"] = theta_uniform
            current_state["q"] = q_uniform
            update_displays(theta_uniform, title_suffix="(Uniform)")

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------
    def on_slider_change(val: str) -> None:
        solve_current()

    def on_scenario_change() -> None:
        scen = scenario_var.get()
        print(f"[GUI] Operating scenario switched to: {scen}")
        solve_current()

    def on_view_type_change() -> None:
        """Handle view type switcher changes."""
        mode_str = view_type_var.get()
        print(f"[GUI] View mode: {mode_str}")
        rebuild_view_canvas()

    def on_animation_change() -> None:
        """Handle animation mode changes."""
        mode = animation_var.get()
        if mode == "Transient Playback":
            on_animate_transient()
        else:
            current_state["is_animating"] = False
            solve_current()

    def on_toggle_labels() -> None:
        """Show or hide functional block labels."""
        show = show_labels_var.get()
        view_type = view_type_var.get()
        if view_type == "3D Surface":
            # Surface needs re-plotting of texts or rebuilding
            theta_curr = current_state["theta"]
            title_suffix = current_state["title_suffix"]
            update_displays(theta_curr, title_suffix=title_suffix)
        else:
            for t in view_refs["labels"]:
                t.set_visible(show)
            canvas_top.draw_idle()

    def on_run_lp() -> None:
        current_state["is_lp"] = True
        print(f"[GUI] Display switched to LP-Optimal (Peak: {current_state['peak_lp']:.2f} K)")
        solve_current()

    def on_show_uniform() -> None:
        current_state["is_lp"] = False
        print(f"[GUI] Display switched to Uniform Baseline (Peak: {current_state['peak_uniform']:.2f} K)")
        solve_current()

    def on_animate_transient() -> None:
        """Trigger non-blocking transient Backward Euler animation."""
        if current_state["is_animating"]:
            return

        current_state["is_animating"] = True
        animation_var.set("Transient Playback")
        status_var.set("Simulating Transient...")
        root.update_idletasks()

        active_q = current_state["q"]
        fields = transient_solve(block_powers=active_q, dt=5e-3, n_steps=60, A=A, B=B)
        total_frames = len(fields)
        dt_val = 5e-3

        def _play_frame(idx: int) -> None:
            if idx >= total_frames or not current_state["is_animating"]:
                current_state["is_animating"] = False
                status_var.set("Optimal")
                current_state["sim_time"] = f"{total_frames * dt_val:.2f} s (Done)"
                update_displays(fields[-1], title_suffix="(Transient Final)")
                return

            sim_t = idx * dt_val
            current_state["sim_time"] = f"{sim_t:.3f} s"
            update_displays(fields[idx], title_suffix=f"(t = {sim_t:.3f} s)")
            root.after(33, lambda: _play_frame(idx + 1))  # ~30 fps

        _play_frame(0)

    def on_reset() -> None:
        p_budget_var.set(DEFAULT_P_MAX)
        scenario_var.set("Symmetric")
        view_type_var.set("3D Surface")
        animation_var.set("Steady State (Fixed)")
        show_labels_var.set(True)
        current_state["is_lp"] = True
        current_state["is_animating"] = False
        rebuild_view_canvas()
        solve_current()

    # -------------------------------------------------------------------------
    # Middle Column: Interactive Control Widgets
    # -------------------------------------------------------------------------
    # 1. Power Budget Slider
    ttk.Label(middle_frame, text="POWER BUDGET (P_max):", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 2))
    budget_scale = tk.Scale(
        middle_frame,
        from_=1.0,
        to=8.0,
        resolution=0.1,
        orient="horizontal",
        variable=p_budget_var,
        command=on_slider_change,
        length=240,
        font=("Consolas", 10),
    )
    budget_scale.pack(fill="x", pady=(0, 10))

    # 2. Operating Scenario Radio Buttons
    ttk.Label(middle_frame, text="SCENARIO SELECTION:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 2))
    ttk.Radiobutton(
        middle_frame, text="Symmetric (4 Cores Active)", value="Symmetric", variable=scenario_var, command=on_scenario_change
    ).pack(anchor="w", pady=1)
    ttk.Radiobutton(
        middle_frame, text="Asymmetric (Cores 0 & 1 Active)", value="Asymmetric", variable=scenario_var, command=on_scenario_change
    ).pack(anchor="w", pady=(1, 10))

    # 3. View Mode Switcher: Part A — "VIEW TYPE"
    ttk.Label(middle_frame, text="VIEW TYPE:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 2))
    ttk.Radiobutton(
        middle_frame, text="3D Surface", value="3D Surface", variable=view_type_var, command=on_view_type_change
    ).pack(anchor="w", pady=1)
    ttk.Radiobutton(
        middle_frame, text="2D Heatmap", value="2D Heatmap", variable=view_type_var, command=on_view_type_change
    ).pack(anchor="w", pady=1)
    ttk.Radiobutton(
        middle_frame, text="3D Labeled Floorplan", value="3D Labeled Floorplan", variable=view_type_var, command=on_view_type_change
    ).pack(anchor="w", pady=(1, 10))

    # 4. View Mode Switcher: Part B — "ANIMATION"
    ttk.Label(middle_frame, text="ANIMATION:", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 2))
    ttk.Radiobutton(
        middle_frame, text="Steady State (Fixed)", value="Steady State (Fixed)", variable=animation_var, command=on_animation_change
    ).pack(anchor="w", pady=1)
    ttk.Radiobutton(
        middle_frame, text="Transient Playback", value="Transient Playback", variable=animation_var, command=on_animation_change
    ).pack(anchor="w", pady=(1, 10))

    # 5. Core Labels Toggle
    chk_labels = ttk.Checkbutton(
        middle_frame, text="Show block labels", variable=show_labels_var, command=on_toggle_labels
    )
    chk_labels.pack(anchor="w", pady=(0, 10))

    # 6. Action Buttons
    ttk.Separator(middle_frame, orient="horizontal").pack(fill="x", pady=4)

    btn_lp = ttk.Button(middle_frame, text="▶ Run LP Optimizer", command=on_run_lp)
    btn_lp.pack(fill="x", pady=3)

    btn_unif = ttk.Button(middle_frame, text="⚡ Show Uniform Allocation", command=on_show_uniform)
    btn_unif.pack(fill="x", pady=3)

    btn_trans = ttk.Button(middle_frame, text="⏱ Animate Transient (0.5 s)", command=on_animate_transient)
    btn_trans.pack(fill="x", pady=3)

    btn_reset = ttk.Button(middle_frame, text="↺ Reset Parameters", command=on_reset)
    btn_reset.pack(fill="x", pady=(3, 10))

    # 7. Status Indicator
    ttk.Separator(middle_frame, orient="horizontal").pack(fill="x", pady=4)
    status_label_title = ttk.Label(middle_frame, text="OPTIMIZER STATUS:", font=("Segoe UI", 9, "bold"))
    status_label_title.pack(anchor="w", pady=(2, 2))

    status_display = ttk.Label(
        middle_frame,
        textvariable=status_var,
        font=("Consolas", 11, "bold"),
        foreground="#1B5E20",
    )
    status_display.pack(anchor="w", pady=2)

    # -------------------------------------------------------------------------
    # Window Close Handler
    # -------------------------------------------------------------------------
    def on_closing() -> None:
        current_state["is_animating"] = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Initial setup and solve
    rebuild_view_canvas()
    solve_current()

    print("[GUI] Ready. Interact with the window.")
    root.mainloop()
    print("[GUI] Window closed by user.")


if __name__ == "__main__":
    main()
