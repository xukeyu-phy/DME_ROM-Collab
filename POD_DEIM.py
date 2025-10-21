import warnings
import json
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from config import Config
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in matmul')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in matmul')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in matmul')


class POD_DEIM:
    def __init__(self, dataset_dir, outdata_dir, fig_dir, device=None, dtype=torch.float64):
        self.dataset_dir = dataset_dir
        self.outdata_dir = outdata_dir
        self.fig_dir = fig_dir
        self.device = device
        self.dtype = dtype
        torch.set_default_dtype(dtype)
        self.r = 1
        self.rf = 1

    def _main_line(self, recompute=False):
        if recompute:
            print("\n" + "="*60)
            print("COMPUTING POD-DEIM DECOMPOSITION")
            print("="*60)
            snapshots, F_snapshots, _ = self.load_snapshots(compute_nonlinear=True)

            Phi_h, S_h, Vt_h, mean_h = self.perform_POD(snapshots)
            self.save_decomposition("POD", Phi_h, S_h, Vt_h, mean=mean_h)
            self.plot_singular_values(S_h, trun_modes = self.r, save_path=self.fig_dir / "POD_singular_values.png")
            self.print_energy_analysis(S_h, "POD")

            Phi_F, P_F, S_F = self.perform_DEIM(F_snapshots)
            self.save_decomposition("DEIM", Phi_F, S_F, indices=P_F)
            self.plot_singular_values(S_F, trun_modes = self.rf, save_path=self.fig_dir / "DEIM_singular_values.png")
            self.print_energy_analysis(S_F, "DEIM")
        else:
            print("\n" + "="*60)
            print("LOADING EXISTING DECOMPOSITION")
            print("="*60)
            Phi_h, S_h, Vt_h, mean_h, _ = self.load_decomposition("POD", load_mean=True)
            Phi_F, S_F, _, _, P_F = self.load_decomposition("DEIM", load_indices=True)
            self.plot_singular_values(S_h, trun_modes = self.r, save_path=self.fig_dir / "POD_singular_values.png")
            self.plot_singular_values(S_F, trun_modes = self.rf, save_path=self.fig_dir / "DEIM_singular_values.png")
    
        print("\n" + "="*60)
        print("COMPLETED")
        print("="*60)    

    # ======================================================
    # Load data and precompute
    # ======================================================
    def load_snapshots(self, compute_nonlinear=True):
        folders = sorted([f for f in self.dataset_dir.iterdir() if f.is_dir() and f.name.isdigit()],
                         key=lambda x: int(x.name))
        snapshots_list, F_snapshots_list, parameters_list = [], [], []

        print(f"Loading {len(folders)} snapshots from {self.dataset_dir}...")
        for folder in folders:
            with open(folder / "parameters.json", 'r', encoding='utf-8') as f:
                params = json.load(f)
            parameters_list.append(params)
            rho_full = torch.load(folder / "DME_HFresult.pt", map_location='cpu', weights_only=True)
            xi_full = torch.load(folder / "xi.pt", map_location='cpu', weights_only=True)
            rho = rho_full[1:-1, 1:-1, 1:-1, :]
            snapshots_list.append(rho.reshape(-1, 1))

            # Compute the nonlinear term
            if compute_nonlinear:
                config = Config(self.device, self.dtype)
                config._setup_phy_ps(params)
                config._setup_matrices(config.Qa, config.Qb)

                rho_full = rho_full.to(self.device)
                xi_full = xi_full.to(self.device)
                A_op_rho = torch.einsum('ij,klmj->klmi', config.A_op, rho_full)
                F = config.R0 * xi_full * A_op_rho
                F = F[1:-1, 1:-1, 1:-1, :]
                F_snapshots_list.append(F.reshape(-1, 1).cpu())

        snapshots = torch.cat(snapshots_list, dim=1)
        print(f"✓ Snapshots matrix: {snapshots.shape}, {snapshots.element_size() * snapshots.numel() / 1e6:.2f} MB")

        if compute_nonlinear:
            F_snapshots = torch.cat(F_snapshots_list, dim=1)
            print(f"✓ Nonlinear snapshots: {F_snapshots.shape}, {F_snapshots.element_size() * F_snapshots.numel() / 1e6:.2f} MB")
            return snapshots, F_snapshots, parameters_list

        return snapshots, parameters_list

    # ======================================================
    # POD
    # ======================================================
    def perform_POD(self, snapshots, n_modes=None, subtract_mean=True):
        print(f"\n{'='*50}")
        print(f"Performing POD on matrix {snapshots.shape}")
        print(f"{'='*50}")

        t0 = time.time()
        snapshots_np = snapshots.cpu().numpy()

        if subtract_mean:
            mean = np.mean(snapshots_np, axis=1, keepdims=True)
            snapshots_centered = snapshots_np - mean
        else:
            mean = np.zeros_like(snapshots_np)
            snapshots_centered = snapshots_np

        U, S, Vt = np.linalg.svd(snapshots_centered, full_matrices=False)

        if n_modes is not None and n_modes < len(S):
            U, S, Vt = U[:, :n_modes], S[:n_modes], Vt[:n_modes, :]

        print(f"✓ POD completed in {time.time()-t0:.2f}s")
        print(f"  Modes: {len(S)}, Top 5 singular values: {S[:5]}")

        return torch.from_numpy(U), torch.from_numpy(S), torch.from_numpy(Vt), torch.from_numpy(mean)

    # ======================================================
    # DEIM
    # ======================================================
    def perform_DEIM(self, snapshots, n_modes=None, energy_threshold=None):
        print(f"\n{'='*50}")
        print(f"Performing DEIM on matrix {snapshots.shape}")
        print(f"{'='*50}")

        t0 = time.time()
        snapshots_np = snapshots.cpu().numpy()
        U, S, Vt = np.linalg.svd(snapshots_np, full_matrices=False)

        if energy_threshold is not None:
            cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
            n_modes = np.where(cumulative_energy >= 1 - energy_threshold)[0][0] + 1
            print(f"  Energy threshold {energy_threshold} → {n_modes} modes")
        elif n_modes is not None:
            n_modes = n_modes
        else:
            n_modes = len(S)

        Phi_F = U[:, :n_modes]
        P = []
        for k in range(n_modes):
            if k == 0:
                p = np.argmax(np.abs(Phi_F[:, 0]))
            else:
                A, b = Phi_F[P, :k], Phi_F[P, k]
                try:
                    c = np.linalg.lstsq(A, b, rcond=1e-10)[0]
                except:
                    c = np.linalg.pinv(A) @ b
                r = Phi_F[:, k] - Phi_F[:, :k] @ c
                p = np.argmax(np.abs(r)) if np.max(np.abs(r)) >= 1e-12 else np.argmax(np.abs(Phi_F[:, k]))
            P.append(int(p))

        print(f"✓ DEIM completed in {time.time()-t0:.2f}s")
        print(f"  Selected {len(P)} interpolation points")
        return torch.from_numpy(U), torch.tensor(P), torch.from_numpy(S)

    # ======================================================
    # Plot
    # ======================================================
    def plot_singular_values(self, S, energy_threshold=1e-10, trun_modes=5, save_path=None):
        S_np = S.cpu().numpy() if isinstance(S, torch.Tensor) else S
        energy = np.cumsum(S_np**2) / np.sum(S_np**2)
        
        # 修正条件判断逻辑
        if trun_modes is None:
            trun_modes = max(np.where(energy >= 1 - energy_threshold)[0][0] + 1, 5)
        else:
            modes_1 = np.where(energy >= 1 - energy_threshold)[0][0] + 1
            modes_2 = trun_modes
            modes_3 = np.where(S_np <= S_np[0] * 1e-5)[0][0] + 1 if len(np.where(S_np <= S_np[0] * 1e-4)[0]) > 0 else len(S_np)
            trun_modes = max(modes_1, modes_2, modes_3)
        plt_modes = min(50, len(S_np))

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.semilogy(np.arange(1, plt_modes + 1), S_np[:plt_modes], '.-', color='tab:blue', linewidth=1.2, label='Singular values')
        ax1.set_xlabel('Mode index')
        ax1.set_ylabel('Singular value', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        ax2.plot(np.arange(1, plt_modes + 1), energy[:plt_modes], 'x-', color='tab:red', linewidth=1.2, label='Cumulative energy')
        ax2.set_ylabel('Cumulative energy', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        if trun_modes <= plt_modes:
            ax1.axvline(x=trun_modes, color='k', linestyle='--', linewidth=0.5)
            ax1.text(trun_modes + 0.7, 1e-2 * max(S[:plt_modes]),
                    f'r={trun_modes}',
                    rotation=0, verticalalignment='center', horizontalalignment='center')
            trunc_sv = S[trun_modes - 1]
            ax1.text(trun_modes, trunc_sv * 1.5, r'$\sigma = $' + f'{trunc_sv:.2e}', 
                fontsize=9, color='black', ha='center')
            
        ax1.text(1, S[0] * 0.2, r'$\sigma = $' + f'{S[0]:.2e}', 
            fontsize=9, color='black', ha='center')
        
        # plt.title(f'Singular Values')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.set_xticks(np.arange(1, plt_modes + 2, 2))


        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        plt.close()

    def print_energy_analysis(self, S, name="POD"):
        total_energy = (S**2).sum()
        cumulative_energy = torch.cumsum(S**2, dim=0) / total_energy
        print(f"\n{name} Energy Analysis:")
        print("-" * 40)
        for i in [1, 2, 3, 5, 10, 20]:
            if i <= len(S):
                print(f"  Top {i:2d} modes: {cumulative_energy[i-1]:.8f}")


    def save_decomposition(self, prefix, Phi, S, Vt=None, mean=None, indices=None):
        torch.save(Phi, self.outdata_dir / f"{prefix}_Phi.pt")
        torch.save(S, self.outdata_dir / f"{prefix}_S.pt")
        if Vt is not None:
            torch.save(Vt, self.outdata_dir / f"{prefix}_Vt.pt")
        if mean is not None:
            torch.save(mean, self.outdata_dir / f"{prefix}_mean.pt")
        if indices is not None:
            torch.save(indices, self.outdata_dir / f"{prefix}_indices.pt")
        print(f"✓ {prefix} results saved to {self.outdata_dir}")

    def load_decomposition(self, prefix, load_mean=False, load_indices=False):
        Phi = torch.load(self.outdata_dir / f"{prefix}_Phi.pt", weights_only=True)
        S = torch.load(self.outdata_dir / f"{prefix}_S.pt", weights_only=True)
        Vt_path = self.outdata_dir / f"{prefix}_Vt.pt"
        Vt = torch.load(Vt_path, weights_only=True) if Vt_path.exists() else None
        mean = torch.load(self.outdata_dir / f"{prefix}_mean.pt", weights_only=True) if load_mean else None
        indices = torch.load(self.outdata_dir / f"{prefix}_indices.pt", weights_only=True) if load_indices else None
        return Phi, S, Vt, mean, indices

