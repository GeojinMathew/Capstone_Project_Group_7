"""Single entry point for the RISC-V Thermal-Aware Power Budgeting pipeline.

Executes the full pipeline end-to-end:
[1/6] Assembles discrete 5-point Laplacian A and power matrix B.
[2/6] Performs analytical modal verification against sparse direct solver.
[3/6] Evaluates Scenario A (symmetric: all 4 cores active).
      Generates 2D heatmaps and 3D surface plots.
[4/6] Evaluates Scenario B (asymmetric: cores 0 & 1 active, cores 2 & 3 idle).
      Generates 2D heatmaps, bar charts, 3D surfaces, 3D extruded floorplan, and rotating GIF.
[5/6] Performs parametric power budget sweep (1 W to 8 W) for Scenario B.
[6/6] Displays final comparative summary table.
"""

from pathlib import Path
import sys
import numpy as np

# Add local directory to python path for standalone execution
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.floorplan import P_max, T_amb
from src.modal_verification import verify_against_sparse
from src.optimizer import solve_lp
from src.power_model import block_powers
from src.sweep import sweep_power_budget
from src.thermal_solver import build_A, build_B, solve_from_regions
from src.visualize import (
    plot_3d_floorplan,
    plot_3d_surface,
    plot_heatmap,
    plot_power_bars,
    plot_rotating_3d,
    plot_sweep,
)


def main() -> None:
    figures_dir = current_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("RISC-V QUAD-CORE THERMAL-AWARE POWER BUDGETING PIPELINE")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # [1/6] Build finite-difference system matrices
    # -------------------------------------------------------------------------
    print("\n[1/6] Assembling finite-difference conductance matrix A and power matrix B...")
    A = build_A()
    B = build_B()
    print(f"      Matrix A shape: {A.shape}, nonzeros (nnz): {A.nnz}, format: {A.format}")
    print(f"      Matrix B shape: {B.shape}, dtype: {B.dtype}")

    # -------------------------------------------------------------------------
    # [2/6] Modal Verification
    # -------------------------------------------------------------------------
    print("\n[2/6] Running analytical modal verification (eigenfunction expansion)...")
    rel_error = verify_against_sparse(A)
    print(f"      Modal verification completed successfully (relative error = {rel_error:.4e} < 1e-10).")

    # -------------------------------------------------------------------------
    # [3/6] Scenario A: Symmetric (All 4 cores active)
    # -------------------------------------------------------------------------
    print("\n[3/6] Evaluating SCENARIO A: Symmetric (all 4 cores active)...")
    cores_A = (0, 1, 2, 3)
    raw_p_A = block_powers(active_cores=cores_A)
    raw_q_A = np.array([raw_p_A["core0"], raw_p_A["core1"], raw_p_A["core2"], raw_p_A["core3"], raw_p_A["l2"]])
    q_uniform_A = raw_q_A * (P_max / np.sum(raw_q_A))

    theta_uniform_A = solve_from_regions(q_uniform_A, A, B)
    peak_uniform_A = float(T_amb + np.max(theta_uniform_A))

    # Solve LP
    lp_res_A = solve_lp(A, B, P_max=P_max)
    q_lp_A = lp_res_A["q"]
    theta_lp_A = lp_res_A["theta"]
    peak_lp_A = float(T_amb + lp_res_A["t_peak"])
    reduction_A = peak_uniform_A - peak_lp_A

    print(f"      Uniform Peak Temperature: {peak_uniform_A:.2f} K")
    print(f"      LP-Optimal Peak Temperature: {peak_lp_A:.2f} K")
    print(f"      Peak Reduction: {reduction_A:.2f} K")
    print(f"      LP Block Powers [core0..core3, L2]: {np.round(q_lp_A, 3).tolist()} W")

    vmin_A = min(np.min(theta_uniform_A), np.min(theta_lp_A)) + T_amb
    vmax_A = max(np.max(theta_uniform_A), np.max(theta_lp_A)) + T_amb

    # 2D Heatmaps
    plot_heatmap(
        theta_uniform_A,
        "Scenario A (Symmetric) - Uniform Allocation",
        figures_dir / "fig_case1_uniform.png",
        vmin=vmin_A,
        vmax=vmax_A,
    )
    plot_heatmap(
        theta_lp_A,
        "Scenario A (Symmetric) - LP-Optimal Allocation",
        figures_dir / "fig_case1_lp.png",
        vmin=vmin_A,
        vmax=vmax_A,
    )

    # 3D Visualizations for Scenario A
    f_c1_u_3d = figures_dir / "fig_case1_uniform_3d.png"
    plot_3d_surface(theta_uniform_A, "Scenario A (Symmetric) - Uniform 3D Surface", f_c1_u_3d)
    print(f"[3D] saved {f_c1_u_3d.name}")

    f_c1_lp_3d = figures_dir / "fig_case1_lp_3d.png"
    plot_3d_surface(theta_lp_A, "Scenario A (Symmetric) - LP-Optimal 3D Surface", f_c1_lp_3d)
    print(f"[3D] saved {f_c1_lp_3d.name}")

    # -------------------------------------------------------------------------
    # [4/6] Scenario B: Asymmetric (Cores 0 & 1 active, Cores 2 & 3 idle)
    # -------------------------------------------------------------------------
    print("\n[4/6] Evaluating SCENARIO B: Asymmetric (cores 0 & 1 active, cores 2 & 3 idle)...")
    cores_B = (0, 1)
    raw_p_B = block_powers(active_cores=cores_B)
    raw_q_B = np.array([raw_p_B["core0"], raw_p_B["core1"], raw_p_B["core2"], raw_p_B["core3"], raw_p_B["l2"]])
    q_uniform_B = raw_q_B * (P_max / np.sum(raw_q_B))

    theta_uniform_B = solve_from_regions(q_uniform_B, A, B)
    peak_uniform_B = float(T_amb + np.max(theta_uniform_B))

    # Solve LP
    lp_res_B = solve_lp(A, B, P_max=P_max)
    q_lp_B = lp_res_B["q"]
    theta_lp_B = lp_res_B["theta"]
    peak_lp_B = float(T_amb + lp_res_B["t_peak"])
    reduction_B = peak_uniform_B - peak_lp_B

    print(f"      Uniform Peak Temperature: {peak_uniform_B:.2f} K")
    print(f"      LP-Optimal Peak Temperature: {peak_lp_B:.2f} K")
    print(f"      Peak Reduction: {reduction_B:.2f} K")
    print(f"      LP Block Powers [core0..core3, L2]: {np.round(q_lp_B, 3).tolist()} W")

    vmin_B = min(np.min(theta_uniform_B), np.min(theta_lp_B)) + T_amb
    vmax_B = max(np.max(theta_uniform_B), np.max(theta_lp_B)) + T_amb

    # 2D Heatmaps and Bar Chart
    plot_heatmap(
        theta_uniform_B,
        "Scenario B (Asymmetric) - Uniform Allocation",
        figures_dir / "fig_case2_uniform.png",
        vmin=vmin_B,
        vmax=vmax_B,
    )
    plot_heatmap(
        theta_lp_B,
        "Scenario B (Asymmetric) - LP-Optimal Allocation",
        figures_dir / "fig_case2_lp.png",
        vmin=vmin_B,
        vmax=vmax_B,
    )
    plot_power_bars(
        q_uniform_B,
        q_lp_B,
        figures_dir / "fig_case2_bars.png",
    )

    # 3D Visualizations for Scenario B
    f_c2_u_3d = figures_dir / "fig_case2_uniform_3d.png"
    plot_3d_surface(theta_uniform_B, "Scenario B (Asymmetric) - Uniform 3D Surface", f_c2_u_3d)
    print(f"[3D] saved {f_c2_u_3d.name}")

    f_c2_lp_3d = figures_dir / "fig_case2_lp_3d.png"
    plot_3d_surface(theta_lp_B, "Scenario B (Asymmetric) - LP-Optimal 3D Surface", f_c2_lp_3d)
    print(f"[3D] saved {f_c2_lp_3d.name}")

    f_c2_floor = figures_dir / "fig_case2_floorplan_3d.png"
    plot_3d_floorplan(theta_lp_B, "Scenario B (Asymmetric) - LP-Optimal 3D Extruded Floorplan", f_c2_floor)
    print(f"[3D] saved {f_c2_floor.name}")

    f_c2_gif = figures_dir / "fig_case2_rotating.gif"
    plot_rotating_3d(theta_lp_B, f_c2_gif, n_frames=60)
    print(f"[3D] saved {f_c2_gif.name}")

    # -------------------------------------------------------------------------
    # [5/6] Parametric Power Budget Sweep
    # -------------------------------------------------------------------------
    print("\n[5/6] Running power budget sweep (1 W to 8 W in 15 steps) for Scenario B...")
    P_array, peak_sweep_u, peak_sweep_lp = sweep_power_budget(active_cores=cores_B)
    plot_sweep(
        P_array,
        peak_sweep_u,
        peak_sweep_lp,
        figures_dir / "fig_sweep.png",
    )
    slope_u = (peak_sweep_u[-1] - peak_sweep_u[0]) / (P_array[-1] - P_array[0])
    slope_lp = (peak_sweep_lp[-1] - peak_sweep_lp[0]) / (P_array[-1] - P_array[0])
    print(f"      Sweep completed. Scaling slopes: Uniform = {slope_u:.2f} K/W, LP = {slope_lp:.2f} K/W.")

    # -------------------------------------------------------------------------
    # [6/6] Final Summary Table
    # -------------------------------------------------------------------------
    print("\n[6/6] Pipeline Execution Finished Successfully.")
    print("=" * 88)
    print("FINAL SUMMARY RESULTS TABLE")
    print("=" * 88)
    header = f"{'Scenario':<12} | {'Uniform Peak (K)':<16} | {'LP Peak (K)':<12} | {'Reduction (K)':<14} | {'LP powers (W)':<30}"
    print(header)
    print("-" * 88)

    str_powers_A = ", ".join(f"{x:.3f}" for x in q_lp_A)
    row_A = f"{'Scenario A':<12} | {peak_uniform_A:<16.2f} | {peak_lp_A:<12.2f} | {reduction_A:<14.2f} | [{str_powers_A}]"
    print(row_A)

    str_powers_B = ", ".join(f"{x:.3f}" for x in q_lp_B)
    row_B = f"{'Scenario B':<12} | {peak_uniform_B:<16.2f} | {peak_lp_B:<12.2f} | {reduction_B:<14.2f} | [{str_powers_B}]"
    print(row_B)
    print("=" * 88)


if __name__ == "__main__":
    main()
