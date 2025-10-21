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
    - Compute the coupled PDE $\xi(z)$ using BDF2, Richardson extrapolation, RK2, and integral methods
    - Comprehensive visualization tools
- ROM
    - Affine parameter dependence
    - NQE (nonlinear quadratic expansion)
    - DEIM
    - Performance acceleration

## 3. [Change Log](https://github.com/xukeyu-phy/Diffusion_Master_Equation/blob/main/CHANGELOG.md)
Please read **CHANGELOG.md** for the latest updates.


## 4. Usage
### 4.1 Requirements
- Python 3.8+
- Required packages:

  ```bash
  torch (PyTorch) >= 2.0
  numpy >= 1.20
  matplotlib >= 3.5
  pathlib (standard library)
  scipy
  ```


### 4.2 Running the FOM
```bash
python DME_main.py
```

This will:

1. Generate a non-uniform or equal-ratio 3D grid
2. Run the time evolution
3. Save results to `Out_data/` directory

*Note:* We recommend using RK2 to calculate.


### 4.3 Running the ROM
```bash
python POD_main.py
python ROM_main.py
```
*Note:* Requires pre-existing data files in `DME_dataset/` from FOM.

### 4.4 Configuration
Modify `config.py` to adjust parameters.


## Permission
***Important Notice***

This project is provided under MIT License with additional restrictions:
- Commercial use prohibited
- Redistribution and modification not permitted without explicit permission
- For academic research use only
- Contact author for collaboration: xukeyu@csrc.ac.cn


