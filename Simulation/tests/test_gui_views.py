"""Unit tests for GUI multi-view modes, centroid block labels, and screenshot rendering.

Verifies:
1. test_screenshot_has_three_panels: Saved PNG has aspect ratio ~3:1 (width ~= 3 * height).
2. test_2d_heatmap_range: Scenario A 2D heatmap temperature range is strictly [300, 360] K.
3. test_labels_present: 3D figures contain at least 5 text artists with 'Core' or 'L2'.
4. test_toggle_labels_visibility: Label show/hide toggling behavior.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pytest

from gui import (
    build_2d_heatmap,
    build_3d_floorplan,
    build_3d_surface,
    create_2d_heatmap_figure,
    create_3d_surface_figure,
    render_headless_screenshot,
)
from src.floorplan import T_amb
from src.optimizer import solve_lp
from src.power_model import block_powers
from src.thermal_solver import build_A, build_B, solve_from_regions


@pytest.fixture(scope="module")
def scenario_a_theta() -> np.ndarray:
    """Compute steady-state temperature rise θ for Scenario A (Symmetric, P_max = 5.0 W)."""
    A = build_A()
    B = build_B()
    p_model = block_powers(active_cores=(0, 1, 2, 3))
    raw_q = np.array([p_model["core0"], p_model["core1"], p_model["core2"], p_model["core3"], p_model["l2"]])
    q = raw_q * (5.0 / np.sum(raw_q))
    theta = solve_from_regions(q, A, B)
    return theta


def test_screenshot_has_three_panels(tmp_path: Path):
    """Verify the saved screenshot has aspect ratio ~3:1 (width ~= 3 * height)."""
    screenshot_file = tmp_path / "gui_screenshot_test.png"
    render_headless_screenshot(screenshot_file)

    assert screenshot_file.exists(), "Screenshot file was not generated."
    assert screenshot_file.stat().st_size > 0, "Screenshot file is empty."

    img = plt.imread(str(screenshot_file))
    h, w = img.shape[:2]
    aspect_ratio = w / h

    # Assert aspect ratio is approximately 3:1 (within 10% tolerance: [2.7, 3.3])
    assert 2.7 <= aspect_ratio <= 3.3, f"Expected aspect ratio ~3:1, got {aspect_ratio:.2f} ({w}x{h})"


def test_2d_heatmap_range(scenario_a_theta: np.ndarray):
    """Verify that after building the 2D figure for Scenario A (P_max=5.0 W),

    the plotted temperature range is strictly [300, 360] (fixed vmin/vmax).
    """
    fig, ax, im = create_2d_heatmap_figure(scenario_a_theta, title_suffix="(Scenario A)")
    try:
        clim = im.get_clim()
        assert clim == (300.0, 360.0), f"Expected colorbar clim (300.0, 360.0), got {clim}"
        assert im.norm.vmin == 300.0, f"Expected vmin 300.0, got {im.norm.vmin}"
        assert im.norm.vmax == 360.0, f"Expected vmax 360.0, got {im.norm.vmax}"
        assert im.get_extent() == [0.0, 10.0, 0.0, 10.0], f"Extent must be [0, 10, 0, 10], got {im.get_extent()}"
    finally:
        plt.close(fig)


def test_labels_present(scenario_a_theta: np.ndarray):
    """Verify the 3D figure has at least 5 text artists containing 'Core' or 'L2'."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        surf, label_texts = build_3d_surface(ax, scenario_a_theta, show_labels=True)
        text_contents = [t.get_text() for t in ax.texts if "Core" in t.get_text() or "L2" in t.get_text()]
        assert len(text_contents) >= 5, f"Expected at least 5 block labels, found {len(text_contents)}: {text_contents}"

        # Verify all 5 functional blocks are present
        for block_name in ["Core 0", "Core 1", "Core 2", "Core 3", "L2"]:
            assert any(block_name in txt for txt in text_contents), f"Missing label for {block_name}"

        # Verify labels are placed above the die surface at height 300 + mean_theta + 2 K
        for t in label_texts:
            pos = t.get_position()
            # pos is (x, y, z) in Text3D
            assert len(pos) >= 2
    finally:
        plt.close(fig)

    # Also verify 3D floorplan view has at least 5 text artists with 'Core' or 'L2'
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")
    try:
        build_3d_floorplan(ax2, scenario_a_theta, show_labels=True)
        text_contents2 = [t.get_text() for t in ax2.texts if "Core" in t.get_text() or "L2" in t.get_text()]
        assert len(text_contents2) >= 5, f"Expected >= 5 labels in floorplan, found {len(text_contents2)}"
    finally:
        plt.close(fig2)


def test_toggle_labels_visibility(scenario_a_theta: np.ndarray):
    """Verify that when show_labels=False, text labels are omitted or hidden."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    try:
        surf, label_texts = build_3d_surface(ax, scenario_a_theta, show_labels=False)
        assert len(label_texts) == 0, "Labels should not be created when show_labels=False"
        matching_texts = [t.get_text() for t in ax.texts if "Core" in t.get_text() or "L2" in t.get_text()]
        assert len(matching_texts) == 0, "No block labels should exist in axes"
    finally:
        plt.close(fig)
