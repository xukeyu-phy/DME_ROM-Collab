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
版权所有 (c) 2025 [xukeyu]
保留所有权利。

本项目及相关文档文件（以下简称“项目”）仅限于学术评审和合作研究使用。
未经作者明确书面许可，任何个人或组织不得：

1. 将项目用于商业用途
2. 修改、复制、分发本项目的任何部分
3. 基于本项目创作衍生作品
4. 将项目用于生产环境或实际部署

当前项目版本为研究预览版，论文工作仍在进行中。
如需获取完整使用权限或合作机会，请联系：[xukeyu@csrc.ac.cn]

本项目按"原样"提供，不提供任何明示或暗示的担保。
作者不对因使用本项目而产生的任何损失或损害负责。


重要声明：本项目为研究原型，可能存在错误和不完整功能
