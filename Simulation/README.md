# Thermal-Aware Power Budgeting for a Quad-Core RISC-V Processor
> **A Full-Stack Pipeline from Architecture to Linear-Programming Optimization**  
> *Course: PDEs, Transforms and Optimization Techniques — B.Tech Capstone*

---

## 🚀 Quick Start (One-Command Reproduction)

```bash
pip install -r requirements.txt && python run.py
```

To run the automated verification and test suite:
```bash
python -m pytest tests/ -v
```

Runtime is under 15 seconds on a standard laptop CPU.

---

## 🖥️ Live Simulator
Run `python gui.py` (or `python run_gui.py`) to launch an interactive dashboard with:
- Real-time 3D temperature surface
- Sliders for power budget and scenario selection
- Live LP optimization
- Transient heat-spreading animation

> [!NOTE]
> Tkinter is included in standard Python distributions on Windows and macOS. If running on Linux without Tkinter installed:
> ```bash
> sudo apt install python3-tk       # Debian/Ubuntu
> sudo dnf install python3-tkinter  # Fedora
> ```
> For headless environments, run `python gui.py --screenshot` to render `figures/gui_screenshot.png`.

---

## 📌 Project Overview & Architecture

This repository implements an end-to-end mathematical and engineering pipeline that:
1. **Models a 4-core RISC-V processor floorplan** on a $10\text{ mm} \times 10\text{ mm}$ silicon die ($t_d = 150\,\mu\text{m}$, $k = 150\text{ W/(m}\cdot\text{K)}$).
2. **Generates spatially resolved power maps** for 5 functional blocks (Core 0, Core 1, Core 2, Core 3, and a shared central L2 cache) with active ($1.2\text{ W}$) and idle ($0.3\text{ W}$) power modes.
3. **Solves the steady-state Poisson heat equation** with a 5-point finite-difference stencil on a $40 \times 40$ interior grid ($N^2 = 1600$ nodes) with $300\text{ K}$ Dirichlet boundary conditions.
4. **Formulates and solves a Linear Program (LP)** in epigraph form that redistributes a strict $5.0\text{ W}$ power budget to minimize the peak die temperature.
5. **Performs modal verification** comparing the finite-difference solver against analytical eigenfunction expansions (matching to machine precision $< 10^{-10}$).
6. **Produces publication-grade figures**: heatmaps, power distribution bar charts, and power budget scaling sweeps.

---

## 📐 Mathematical Formulation

### 1. Governing PDE & Boundary Conditions
From Fourier's law ($\mathbf{q}'' = -k \nabla T$) and energy balance at steady state:
$$\nabla^2 T = -\frac{Q(x, y)}{k}, \quad T = 300\text{ K on } \partial\Omega$$
In terms of temperature rise $\theta = T - 300\text{ K}$:
$$\nabla^2 \theta = -\frac{Q}{k}, \quad \theta = 0\text{ on } \partial\Omega$$

### 2. Discretization (5-Point Stencil)
Conductive nodal balance on cell $(i, j)$ with grid spacing $h = L/(N+1)$:
$$k \cdot t_d \left(4 \theta_{i,j} - \theta_{i+1,j} - \theta_{i-1,j} - \theta_{i,j+1} - \theta_{i,j-1}\right) = q_{i,j}$$
Assembled into a sparse linear system:
$$A \theta = B q$$
where:
- $A = k \cdot t_d (I \otimes D + D \otimes I)$ with $D = \text{tridiag}(-1, 2, -1)$, size $1600 \times 1600$, symmetric positive-definite.
- $B \in \mathbb{R}^{1600 \times 5}$ maps block powers $q \in \mathbb{R}^5$ to nodal power injections.

### 3. Minimax Linear Program (Epigraph Form)
$$\begin{aligned}
\min_{q, \theta, t} \quad & t \\
\text{s.t.} \quad & A \theta - B q = 0 \\
& -t \le \theta_i \le t, \quad i = 1, \dots, 1600 \\
& \sum_{j=1}^5 q_j = P_{\max} = 5.0\text{ W} \\
& q_j \ge 0, \quad t \ge 0
\end{aligned}$$
Decision vector: $x = [q\ (5), \theta\ (1600), t\ (1)] \in \mathbb{R}^{1606}$. Solved with `scipy.optimize.linprog(method='highs')`.

### 4. Analytical Modal Verification
Eigenfunctions of $-\nabla^2$ on $[0, L]^2$ with zero Dirichlet BC:
$$\varphi_{mn}(x, y) = \frac{2}{L} \sin\left(\frac{m\pi x}{L}\right) \sin\left(\frac{n\pi y}{L}\right), \quad \lambda_{mn} = \frac{\pi^2(m^2 + n^2)}{L^2}$$
Thermal resistance:
$$R_{th}(x, y; x_0, y_0) = \sum_{m,n \ge 1} \frac{\varphi_{mn}(x_0, y_0) \varphi_{mn}(x, y)}{k \cdot t_d \cdot \lambda_{mn}}$$
Matches the discrete sparse direct solver to machine precision ($< 10^{-10}$).

---

## 📊 Summary of Results ($P_{\max} = 5.0\text{ W}$)

| Scenario | Operating State | Uniform Peak (K) | LP Peak (K) | Reduction (K) | LP Powers $q_0 \dots q_4$ (W) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Scenario A** | All 4 cores active (symmetric) | 325.18 | 322.86 | **2.31** | `[1.175, 1.175, 1.175, 1.175, 0.300]` |
| **Scenario B** | Cores 0 & 1 active, 2 & 3 idle | 328.65 | 322.86 | **5.79** | `[1.175, 1.175, 1.175, 1.175, 0.300]` |

### Physical Insights
- **Thermal Heat-Sink Proximity**: The 4 cores reside adjacent to the cooler ambient perimeter ($x, y$ boundaries), whereas the central L2 cache has the longest conduction distance to the sink.
- **Architectural Lower Bound**: With realistic lower bounds ($q_{\text{L2}} \ge 0.30\text{ W}$, $q_{\text{core}} \ge 0.10\text{ W}$), the LP allocates $0.300\text{ W}$ to L2 and shares the remaining $4.700\text{ W}$ evenly across the four cores ($1.175\text{ W}$ each), lowering peak temperature by up to **5.79 K** in Scenario B.
- **Linear Scaling**: Peak temperature rise scales linearly with $P_{\max}$ ($8.37\text{ K/W}$ under LP optimization vs. $10.40\text{ K/W}$ under uniform power).

---

## 📁 Repository Structure

```
riscv_thermal/
├── README.md
├── requirements.txt
├── run.py                          # Single pipeline entry point (batch)
├── gui.py                          # Live Tkinter interactive simulator dashboard
├── run_gui.py                      # Launcher script for live simulator GUI
├── src/
│   ├── __init__.py
│   ├── floorplan.py                # Die constants, REGIONS dict, coordinate masks
│   ├── power_model.py              # Active/idle power models and power map generator
│   ├── thermal_solver.py           # Sparse matrix builders (A, B) and PDE solver
│   ├── modal_verification.py       # Analytical modal series verification
│   ├── optimizer.py                # HiGHS Linear Program (epigraph formulation)
│   ├── visualize.py                # Publication 2D & 3D heatmaps, surfaces, and animations
│   ├── transient.py                # Backward Euler time-stepping solver & GUI player
│   └── sweep.py                    # Parametric power budget sweep (1 to 8 W)
├── data/
│   └── .gitkeep
├── figures/
│   ├── fig_case1_uniform.png       # Scenario A uniform 2D heatmap
│   ├── fig_case1_lp.png            # Scenario A LP-optimal 2D heatmap
│   ├── fig_case1_uniform_3d.png    # Scenario A uniform 3D surface plot
│   ├── fig_case1_lp_3d.png         # Scenario A LP-optimal 3D surface plot
│   ├── fig_case2_uniform.png       # Scenario B uniform 2D heatmap
│   ├── fig_case2_lp.png            # Scenario B LP-optimal 2D heatmap
│   ├── fig_case2_uniform_3d.png    # Scenario B uniform 3D surface plot
│   ├── fig_case2_lp_3d.png         # Scenario B LP-optimal 3D surface plot
│   ├── fig_case2_floorplan_3d.png  # Scenario B extruded 3D floorplan view
│   ├── fig_case2_rotating.gif      # Scenario B animated rotating 3D surface
│   ├── fig_case2_bars.png          # Grouped power allocation bar chart
│   ├── fig_sweep.png               # Peak temperature vs budget sweep
│   └── gui_screenshot.png          # Live simulator dashboard screenshot
└── tests/
    ├── test_thermal.py             # Symmetry, positive definiteness, modal & 3D check
    ├── test_lp.py                  # LP optimality, budget equality, symmetry check
    └── test_transient.py           # Convergence, monotonicity, and speed checks
```
