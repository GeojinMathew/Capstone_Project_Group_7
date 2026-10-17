# Capstone Project: Thermal-Aware Power Budgeting for a Quad-Core RISC-V Processor

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-16%20Passed%20%7C%20100%25-brightgreen.svg)]()
[![Optimization](https://img.shields.io/badge/Optimization-HiGHS%20Linear%20Program-orange.svg)]()
[![Affiliation](https://img.shields.io/badge/Institution-Saintgits%20College%20of%20Engineering-purple.svg)]()

An end-to-end mathematical, numerical, and computational pipeline for **thermal-aware power budgeting of a quad-core RISC-V microprocessor** on a $10\text{ mm} \times 10\text{ mm}$ silicon die.

Developed for the course **PDEs, Transforms and Optimization Techniques** by **Capstone Project Group 7**, Department of Electronics and Communication Engineering (ECE), Saintgits College of Engineering.

---

## 📌 1. Project Overview

Uneven workload and power dissipation in modern multi-core microprocessors lead to severe localized thermal hotspots. Hotspots degrade carrier mobility, exacerbate interconnect electromigration, accelerate dielectric breakdown, and increase leakage currents.

This project formulates an end-to-end mathematical and computational framework that:
1. **Models heat transfer** across a $10\text{ mm} \times 10\text{ mm} \times 150\,\mu\text{m}$ silicon die with 5 functional blocks (Core 0, Core 1, Core 2, Core 3, and a central shared L2 cache).
2. **Solves the steady-state 2D Poisson heat conduction equation** using a 5-point finite-difference stencil on a $40 \times 40$ discretization grid ($N^2 = 1600$ nodes) subject to $300\text{ K}$ Dirichlet boundary conditions.
3. **Analytically validates the numerical solver** against an eigenfunction expansion (double Fourier sine series) to machine precision ($\|T_{\text{FD}} - T_{\text{modal}}\|_\infty < 10^{-10}\text{ K}$).
4. **Formulates an epigraph Linear Program (LP)** using SciPy's HiGHS solver to redistribute a strict $5.0\text{ W}$ power budget among active cores, provably minimizing the peak chip temperature while respecting floorplan power bounds.
5. **Simulates transient thermal dynamics** using an implicit Backward Euler scheme with spatial capacitance to model workload transitions.
6. **Delivers an interactive GUI dashboard** with real-time 3D temperature surfaces, parametric budget sweeps, and publication-ready figures.

---

## 🎯 2. Objectives & Problem Statement

### Problem Statement
Standard static or uniform power budgeting allocates power regardless of spatial heat spreading. Due to chip boundary heat sinks, central functional blocks (such as the L2 cache) experience reduced heat dissipation pathways compared to corner or edge blocks. A thermal-oblivious allocation produces severe hotspot temperatures that trigger aggressive thermal throttling and reduce circuit lifespan.

### Key Objectives
- **Thermal Modelling:** Formulate governing boundary value problems (BVPs) from Fourier's law and steady-state energy conservation.
- **Analytical & Modal Solutions:** Derive exact 2D Fourier sine series solutions to verify numerical stability and accuracy.
- **Numerical Discretization:** Assemble high-performance sparse Laplacian conduction matrices $\mathbf{A} \in \mathbb{R}^{1600 \times 1600}$.
- **Constrained Optimization:** Formulate the min-max temperature problem as an epigraph LP to minimize $t \ge T_i$ subject to total budget $\sum P_k \le P_{\text{budget}}$ and block bounds $P_k^{\min} \le P_k \le P_k^{\max}$.
- **Dynamic Analysis:** Implement implicit time-stepping to track thermal time constants ($\tau \approx 4\text{–}6\text{ ms}$).
- **Interactive Tooling & Visualization:** Provide full-stack CLI and GUI tools alongside comprehensive documentation.

---

## 📐 3. Mathematical Methodology

### 3.1 Governing Partial Differential Equations
From Fourier's law ($\mathbf{q}'' = -k \nabla T$) and the continuity of heat flux:
$$\nabla^2 T(x, y) = -\frac{Q(x, y)}{k}, \quad (x, y) \in \Omega = [0, L] \times [0, W]$$
Subject to isothermal Dirichlet boundary conditions:
$$T(x, y) = T_{\text{ambient}} = 300\text{ K on } \partial\Omega$$

Defining temperature rise $\theta(x, y) = T(x, y) - 300\text{ K}$:
$$\nabla^2 \theta = -\frac{Q(x, y)}{k}, \quad \theta\big|_{\partial\Omega} = 0$$

### 3.2 Finite-Difference Discretization
Using a uniform mesh with grid spacing $h = L / (N + 1)$:
$$\frac{\theta_{i+1, j} + \theta_{i-1, j} + \theta_{i, j+1} + \theta_{i, j-1} - 4\theta_{i, j}}{h^2} = -\frac{Q_{i, j}}{k}$$
Vectorized into the sparse linear system:
$$\mathbf{A} \boldsymbol{\theta} = \mathbf{b}, \quad \mathbf{A} = \mathbf{I} \otimes \mathbf{T}_{1\text{D}} + \mathbf{T}_{1\text{D}} \otimes \mathbf{I}$$

### 3.3 Linear Programming Optimization (Epigraph Min-Max)
To minimize the peak temperature rise $\max_i \theta_i$:
$$\min_{t, \, \mathbf{P}} \quad t$$
$$\text{subject to:} \quad \mathbf{G} \mathbf{P} - t \mathbf{1} \le \mathbf{0}$$
$$\sum_{k=1}^K P_k \le P_{\text{budget}}$$
$$P_k^{\min} \le P_k \le P_k^{\max}, \quad \forall k \in \{0, \dots, K-1\}$$
where $\mathbf{G} = \mathbf{A}^{-1} \mathbf{M} \in \mathbb{R}^{N^2 \times K}$ is the precomputed thermal sensitivity matrix mapping unit power in block $k$ to temperature rises across all grid nodes.

---

## 📁 4. Repository Structure

```
Capstone_Project_Group_7/
├── README.md                                                          # Project documentation & execution guide
├── .gitignore                                                         # Git ignore rules for Python / IDE artifacts
├── Capstone_Presentation_Group_7.pptx                                 # Final project presentation slide deck
├── Thermal_Aware_Power_Budgeting_for_a_Quad_Core_RISC_V_Processor.pdf # Comprehensive final project technical report
└── Simulation/                                                        # Full simulation codebase, models, tests & figures
    ├── README.md                  # In-depth mathematical formulation and manual
    ├── requirements.txt           # Python dependencies (numpy, scipy, matplotlib, sympy, pytest)
    ├── run.py                     # Automated 6-stage end-to-end pipeline runner
    ├── gui.py                     # Interactive Tkinter multi-view thermal dashboard
    ├── run_gui.py                 # Convenience launcher for the live GUI
    ├── src/                       # Physical modeling, PDE solver, and optimization modules
    │   ├── floorplan.py           # Physical constants, floorplan geometry & grid discretization
    │   ├── power_model.py         # Functional block power dissipation models
    │   ├── thermal_solver.py      # 2D Poisson finite-difference stencil & sparse conduction matrix
    │   ├── modal_verification.py  # Double Fourier sine series modal verification
    │   ├── optimizer.py           # HiGHS epigraph LP formulation for peak thermal minimization
    │   ├── transient.py           # Backward Euler implicit time-stepping transient solver
    │   ├── visualize.py           # 2D/3D publication visualizations & GIF animations
    │   └── sweep.py               # Parametric budget sweeps
    ├── tests/                     # Automated unit test suite (16 tests, 100% passing)
    │   ├── test_thermal.py        # Physical, stencil, and analytical verification tests
    │   ├── test_lp.py             # LP budget conservation and optimality tests
    │   ├── test_transient.py      # Transient stability and convergence tests
    │   └── test_gui_views.py      # GUI multi-view modes and screenshot aspect ratio tests
    └── figures/                   # Generated publication figures and animations
        ├── gui_screenshot.png     # 3-panel multi-view report figure (3:1 aspect ratio)
        ├── fig_case1_uniform.png  # Scenario A uniform heatmap
        ├── fig_case1_lp.png       # Scenario A LP-optimal heatmap
        ├── fig_case2_uniform.png  # Scenario B uniform heatmap
        ├── fig_case2_lp.png       # Scenario B LP-optimal heatmap
        ├── fig_case2_bars.png     # Power allocation grouped bar chart
        ├── fig_sweep.png          # Peak temperature vs. power budget scaling curve
        ├── fig_case2_rotating.gif # 3D rotating surface animation
        └── ...
```

---

## 🚀 5. Quick Start & Execution Guide

### 5.1 Environment Setup
Ensure Python 3.9+ is installed. Clone the repository and install the dependencies:
```bash
git clone https://github.com/GeojinMathew/Capstone_Project_Group_7.git
cd Capstone_Project_Group_7/Simulation
pip install -r requirements.txt
```

### 5.2 Run the End-to-End Simulation
To execute the automated 6-stage computational pipeline (floorplan discretization, finite-difference matrix assembly, modal analytical verification, LP optimization, transient solver, and figure generation):
```bash
python run.py
```
*Execution takes under 15 seconds on a standard desktop/laptop.*

### 5.3 Launch the Interactive Dashboard
Launch the interactive graphical dashboard with dynamic budget sliders, live LP recalculation, 3D surface view, 2D heatmap view, 3D extruded floorplan view, and transient playback:
```bash
python gui.py
```
*(Or use `python run_gui.py`)*

To generate the headless 3-panel publication figure without opening a display window:
```bash
python gui.py --screenshot
```

### 5.4 Run Automated Unit Tests
Verify physical consistency, solver convergence, LP budget preservation, and GUI views:
```bash
python -m pytest tests/ -v
```
**Results:** `16 passed in ~2.5s (100% passing rate)`.

---

## 📊 6. Key Deliverables & Summary of Results

| Deliverable | Format | Description |
|---|---|---|
| **Technical Report** | `Thermal_Aware_Power_Budgeting_for_a_Quad_Core_RISC_V_Processor.pdf` | Complete academic report covering PDE derivation, stencil assembly, modal analysis, LP theory, transient dynamics, and results. |
| **Presentation Deck** | `Capstone_Presentation_Group_7.pptx` | Presentation slide deck detailing problem motivation, mathematical formulation, optimization results, and conclusions. |
| **Simulation Codebase** | `Simulation/` | Modular, unit-tested, and fully reproducible Python simulation framework and interactive GUI. |
| **Publication Figures** | `Simulation/figures/` | High-resolution heatmaps, 3D surfaces, power allocation bar charts, and transient GIF animations. |

### Summary of Optimization Impact
- **Scenario A (Uniform 5.0 W budget):** Baseline peak temperature rise is reduced by redistributing active power away from poorly cooled central regions to peripheries.
- **Scenario B (Asymmetric Core 0 / Core 3 load):** LP optimization mitigates severe diagonal thermal asymmetry, cutting peak hotspot temperature while maintaining identical total compute power throughput.
- **Verification Precision:** Double Fourier analytical series matches the numerical 5-point Laplacian solution to within $\|T_{\text{FD}} - T_{\text{modal}}\|_\infty < 10^{-10}\text{ K}$.

---

## 👥 7. Team & Academic Affiliation

- **Project Group:** Group 7
- **Course:** PDEs, Transforms and Optimization Techniques
- **Academic Level:** Second Year B.Tech
- **Department:** Department of Electronics and Communication Engineering (ECE)
- **Institution:** Saintgits College of Engineering (Autonomous), Kottayam, Kerala

---

## 📚 8. References

1. Incropera, F. P., DeWitt, D. P., Bergman, T. L., & Lavine, A. S. *Fundamentals of Heat and Mass Transfer*. John Wiley & Sons.
2. Boyd, S., & Vandenberghe, L. *Convex Optimization*. Cambridge University Press.
3. Skadron, K., et al. "Temperature-aware microarchitecture: Modeling and implementation." *ACM Transactions on Computer Systems (TOCS)*, 2003.
4. Huang, W., et al. "HotSpot: A compact thermal modeling methodology for early-stage VLSI design." *IEEE Transactions on VLSI Systems*, 2006.
5. Virtanen, P., et al. "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python." *Nature Methods*, 2020 (HiGHS Optimization Engine).
