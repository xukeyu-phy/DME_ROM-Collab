# DME_ROM-Collab

This project simulates the 3D Diffusion Master Equation using the finite-difference method.

## 1. Overview

### Full Order Model
The Full Order Model (FOM) uses finite-difference methods with Runge–Kutta time integration. It solves coupled partial differential equations describing spin dynamics with optical pumping and diffusion effects.

### Reduced Order Model
The Reduced Order Model (ROM) is intrusive. Hyperreduction methods include DEIM and NQE (nonlinear quadratic expansion).

## 2. Key Features

- FOM
    - 3D non-uniform or equal-ratio grid
    - Second-order Run–Kutta (RK2) time integration
    - Parallel GPU acceleration (CUDA supported)
    - Compute the coupled PDE \(\xi(z)\) using BDF2, Richardson extrapolation, RK2, and integral methods
    - Comprehensive visualization tools
- ROM
    - Affine parameter dependence
    - NQE (nonlinear quadratic expansion)
    - DEIM
    - Performance acceleration

## 3. [Change Log](https://github.com/xukeyu-phy/Diffusion_Master_Equation/blob/main/CHANGELOG.md)
Please read **CHANGELOG.md** for the latest updates.

## 4. Permission
Please read the **license.txt**.

