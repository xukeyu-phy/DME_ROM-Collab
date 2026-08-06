import torch
from config import Config
import time


class ROMSolver:
    def __init__(self, r, rf, Phi_r, Phi_f, P_f, POD_mean, phy_ps, device, dtype, outdata_dir):
        self.device = device
        self.config = Config(device, dtype)
        torch.set_default_dtype(dtype)

        self.r = r
        self.rf = rf
        self.Phi_r = Phi_r.to(device)
        self.Phi_Gop_deim = Phi_f.to(device)
        self.P_f = P_f.to(device)
        self.Nh = Phi_r.shape[0]
        self.N3 = self.Nh // 8
        self.N = round(self.N3 ** (1/3))

        self.config._create_non_uniform_grid(outdata_dir)
        self.config._setup_coefficient()        

        self._init_phy_ps(phy_ps)
        self.POD_mean = POD_mean.to(device)
        rho_mean = self.POD_mean.reshape(self.N, self.N, self.N, 8)
        self.rhs_mean = self.rom_rhs_mean_field(rho_mean)
        self.rom_GNL_preperform_NQE(rho_mean)

        self.nabla_r = self.rom_diffusion_preperform()
        self.G0_r = self.rom_G0_preperform()
        self.rom_DEIM_preperform()


    def _init_phy_ps(self, phy_ps):
        self.config._setup_phy_ps(phy_ps)
        self.config._setup_matrices(self.config.Qa, self.config.Qb)


    def _main_line(self, dt):
        start_time = time.time()
        iter_count = 0
        for iter_count in range(1):
            rho_init = torch.zeros(self.Nh, 1).to(self.device)
            rho_init = self._setup_initial_condition(rho_init)
            alpha_init = self.Phi_r.T @ (
                rho_init - self.POD_mean.reshape(-1, 1)
            )

            t = 0.0
            t_iter = 0
            alpha_n = alpha_init.clone()
            while t < self.config.T_final:
                if (t + dt) >= self.config.T_final:
                    dt = self.config.T_final - t
                t += dt
                t_iter += 1 

                if t_iter % 200 == 1:                                
                    alpha_n0 = alpha_n.clone()
                    alpha_n, residual = self.runge_kutta_1_step(
                        alpha_n, dt, self.config.ghostcell,
                        self.config.bc_type, 'update'
                    )
                    dalpha = torch.abs(alpha_n - alpha_n0)
                    l2_dalpha = torch.norm(dalpha, p=2)
                    l2_residual = torch.norm(torch.norm(residual), p=2)
                    print(f"Iter: {t_iter}, Time: {t:.6f}, l2_dalpha = {l2_dalpha:.4e}")     
                    
                    if l2_residual < self.config.convergence_tol:
                        print(f'Convergence: l2_residual = {l2_residual:.6e}, l2_dalpha = {l2_dalpha:.6e}')
                        break
                else:
                    alpha_n, _ = self.runge_kutta_1_step(
                        alpha_n, dt, self.config.ghostcell,
                        self.config.bc_type, None
                    )

            iter_count += 1        

        end_time = time.time()
        runtime =  end_time - start_time
        print(f"Runtime: {runtime:.3f}s, 1 case time: {runtime/iter_count:3f}")
        # self.print_deim_timing()
        return alpha_n

    def _setup_initial_condition(self, rho):
        if self.config.initial_condition_type == 'uniform':
            rho[:, :] = self.config.initial_value
        elif self.config.initial_condition_type == 'analytical':
            pass        
        return rho

    def rom_diffusion_preperform(self):    
        Phi_space = self.Phi_r.reshape(self.N, self.N, self.N, 8, self.r)
        phi_full = torch.zeros(self.N+2, self.N+2, self.N+2, 8, self.r).to(self.device) 
        phi_full[1:-1, 1:-1, 1:-1, :, :] = Phi_space
        
        APhi = Phi_space.clone()
        self.config.alpha_x = self.config.alpha_x.unsqueeze(-2) 
        self.config.beta_x = self.config.beta_x.unsqueeze(-2)
        self.config.gamma_x = self.config.gamma_x.unsqueeze(-2)
        self.config.alpha_y = self.config.alpha_y.unsqueeze(-2) 
        self.config.beta_y = self.config.beta_y.unsqueeze(-2)
        self.config.gamma_y = self.config.gamma_y.unsqueeze(-2)
        self.config.alpha_z = self.config.alpha_z.unsqueeze(-2) 
        self.config.beta_z = self.config.beta_z.unsqueeze(-2)
        self.config.gamma_z = self.config.gamma_z.unsqueeze(-2)
        APhi_dx = (  self.config.alpha_x * phi_full[:-2, 1:-1, 1:-1, :, :] + 
                     self.config.beta_x * Phi_space[:, :, :, :, :] + 
                     self.config.gamma_x * phi_full[2:, 1:-1, 1:-1, :, :])
        
        APhi_dy = (  self.config.alpha_y * phi_full[1:-1, :-2, 1:-1, :, :] + 
                     self.config.beta_y * Phi_space[:, :, :, :, :] + 
                     self.config.gamma_y * phi_full[1:-1, 2:, 1:-1, :, :])
        
        APhi_dz = (  self.config.alpha_z * phi_full[1:-1, 1:-1, :-2, :, :] + 
                     self.config.beta_z * Phi_space[:, :, :, :, :] + 
                     self.config.gamma_z * phi_full[1:-1, 1:-1, 2:, :, :])
        APhi[:, :, :, :, :] = (APhi_dx + APhi_dy + APhi_dz)    
        APhi = APhi.reshape(-1, self.r)
        nabla_r = self.config.D * self.Phi_r.T @ APhi
        return nabla_r

    def rom_diffusion_mean_preperform(self, rho_mean, bc_value=0.125):     
        rho_full = torch.zeros(self.N+2, self.N+2, self.N+2, 8).to(self.device)
        rho_full[:, :, :, :] = bc_value
        rho_full[1:-1, 1:-1, 1:-1, :] = rho_mean
        Arho_mean = rho_mean.clone()
        Arho_mean_dx = (  self.config.alpha_x * rho_full[:-2, 1:-1, 1:-1, :] + 
                     self.config.beta_x * rho_mean[:, :, :, :] + 
                     self.config.gamma_x * rho_full[2:, 1:-1, 1:-1, :])
        
        Arho_mean_dy = (  self.config.alpha_y * rho_full[1:-1, :-2, 1:-1, :] + 
                     self.config.beta_y * rho_mean[:, :, :, :] + 
                     self.config.gamma_y * rho_full[1:-1, 2:, 1:-1, :])
        
        Arho_mean_dz = (  self.config.alpha_z * rho_full[1:-1, 1:-1, :-2, :] + 
                     self.config.beta_z * rho_mean[:, :, :, :] + 
                     self.config.gamma_z * rho_full[1:-1, 1:-1, 2:, :])
        Arho_mean[:, :, :, :] = (Arho_mean_dx + Arho_mean_dy + Arho_mean_dz)    
        Arho_mean = Arho_mean.reshape(-1, 1)
        diff_mean = self.config.D * self.Phi_r.T @ Arho_mean
        return diff_mean

    def rom_G0_preperform(self): 
        G0_matrix = (1 + self.config.eta) * self.config.A_SD + self.config.fD * self.config.A_FD  
        Phi_r_reshaped = self.Phi_r.reshape(self.N, self.N, self.N, 8, self.r)          
        Tmp = torch.einsum('id,abcde->abcie', G0_matrix, Phi_r_reshaped)
        Tmp_matrix = Tmp.reshape(self.Nh, self.r)
        self.G0_r = torch.matmul(self.Phi_r.T, Tmp_matrix)
        return self.G0_r

    def rom_G0_mean_preperform(self, rho_mean): 
        G0_matrix = (1 + self.config.eta) * self.config.A_SD + self.config.fD * self.config.A_FD  
        Tmp = torch.einsum('id,abcd->abci', G0_matrix, rho_mean)
        Tmp_matrix = Tmp.reshape(self.Nh, 1)          
        G0_mean = torch.matmul(self.Phi_r.T, Tmp_matrix)
        return G0_mean
    
    def rom_rhs_mean_field(self, rho_mean):
        diff_mean = self.rom_diffusion_mean_preperform(rho_mean)
        G0_mean = self.rom_G0_mean_preperform(rho_mean)
        mean = - G0_mean + diff_mean
        return mean    

    def rom_GNL_preperform_NQE(self, rho_mean):
        Phi_r_reshaped = self.Phi_r.reshape(self.N3, 8, self.r)
        S_dot_Phi_r = torch.einsum('b,abc->ac', self.config.S_z, Phi_r_reshaped)
        S_dot_Phi_r = S_dot_Phi_r.unsqueeze(1)
        S_dot_Phi_r = S_dot_Phi_r.expand(-1, 8, -1)
        S_dot_Phi_r = S_dot_Phi_r.reshape(-1, self.r)
        A_SE_Phi_r = torch.einsum('db,abc->adc', self.config.A_SE, Phi_r_reshaped)
        A_SE_Phi_r = A_SE_Phi_r.reshape(-1, self.r)

        A_NQE_Phi_r = torch.zeros(self.Nh, self.r * self.r).to(self.device)
        for i in range(self.r):
            for j in range(self.r):
                A_NQE_Phi_r[:, i * self.r + j] = S_dot_Phi_r[:, i] * A_SE_Phi_r[:, j]
        self.A_NQE = self.Phi_r.T @ A_NQE_Phi_r   

        A_SE_rho = torch.einsum('id,abcd->abci', self.config.A_SE, rho_mean)
        A_SE_rho = A_SE_rho.reshape(-1, 1)
        B_NQE_1 = S_dot_Phi_r * A_SE_rho

        S_dot_rho = torch.einsum('d,abcd->abc', self.config.S_z, rho_mean)
        S_dot_rho = S_dot_rho.unsqueeze(-1)
        S_dot_rho = S_dot_rho.expand(-1, -1, -1, 8)
        S_dot_rho = S_dot_rho.reshape(-1, 1)
        B_NQE_2 = S_dot_rho * A_SE_Phi_r
        self.B_NQE = self.Phi_r.T @ (B_NQE_1 + B_NQE_2)

        C_NQE = S_dot_rho * A_SE_rho
        C_NQE = self.Phi_r.T @ C_NQE
        self.C_NQE_alpha = C_NQE

    def rom_rhs_diffusion(self, alpha):
        return self.nabla_r @ alpha

    def rom_rhs_G0(self, alpha):
        return self.G0_r @ alpha

    def rom_rhs_GNL_NQE(self, alpha):
        alpha_flat = alpha.flatten()
        rho_outer = torch.outer(alpha_flat, alpha_flat)
        rho_outer_flat = rho_outer.flatten().unsqueeze(0)
        A_NQE_alpha = self.A_NQE @ rho_outer_flat.T
        B_NQE_alpha = self.B_NQE @ alpha

        rhs_GNL_r = -self.config.eta * (
            A_NQE_alpha + B_NQE_alpha + self.C_NQE_alpha
        )
        return rhs_GNL_r

    def rhs(self, rho, update=None):
        # The optical-pumping term is evaluated at the current RK stage.
        # ``update`` is retained only for compatibility with ROM_main.py.
        rhs_vib = (
            -self.rom_rhs_G0(rho)
            -self.rom_rhs_Gop_DEIM_paper(rho)
            -self.rom_rhs_GNL_NQE(rho)
            +self.rom_rhs_diffusion(rho)
        )
        rhs = rhs_vib + self.rhs_mean

        return rhs


    def runge_kutta_2_step(self, rho_n, dt, ghostcell=None, bc_type=None, update=None):
        k1 = self.rhs(rho_n, update)
        rho_1 = rho_n + dt * k1

        k2 = self.rhs(rho_1, update)
        rho_new = rho_n + (dt/2.0) * (k1 + k2)
        
        return rho_new


    def runge_kutta_1_step(self, rho_n, dt, ghostcell=None, bc_type=None, update=None):
        k1 = self.rhs(rho_n, update)
        rho_1 = rho_n + dt * k1
        
        return rho_1, k1
    

    def rom_DEIM_preperform(self):
        x, y, z = self.config.grid_x, self.config.grid_y, self.config.grid_z
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)
        self.config.w = self.config.w.to(self.device)

        # Paper notation: V, U_f, P, and rho_bar.
        V = self.Phi_r
        U_f = self.Phi_Gop_deim
        P = self.P_f
        rho_bar = self.POD_mean

        # calD = V^T U_f (P^T U_f)^{-1}
        PTU_f = U_f[P, :]
        self.calD = V.T @ U_f @ torch.linalg.inv(PTU_f)
        # Retain the original class attribute used by earlier ROM code.
        self.GopTmp = self.calD

        # Map each selected scalar index p_s to (i_s, j_s, k_s, q_s).
        # V and rho_bar contain interior nodes only, so i_s, j_s,
        # and k_s below are interior-local indices.
        spatial_indices = torch.div(P, 8, rounding_mode='floor')
        # comp_indices is the zero-based representation of q_s=1,...,8.
        self.comp_indices = P % 8

        self.M = spatial_indices.numel()
        if self.M != self.rf:
            raise ValueError("The DEIM index set P must contain m entries.")

        self.k_idx = (spatial_indices % self.N).to(self.device)
        self.j_idx = ((spatial_indices // self.N) % self.N).to(self.device)
        self.i_idx = (spatial_indices // (self.N * self.N)).to(self.device)

        # Selected physical coordinates and
        # xi_P = exp[-OD*z_s - 2*(x_s^2+y_s^2)/w^2].
        # The +1 shift maps an interior-local index to the full grid,
        # whose index zero is the prescribed inflow boundary.
        self.x_m = x[self.i_idx+1, self.j_idx+1, self.k_idx+1]
        self.y_m = y[self.i_idx+1, self.j_idx+1, self.k_idx+1]
        self.z_m = z[self.i_idx+1, self.j_idx+1, self.k_idx+1]
        self.xi_P = torch.exp(
            -self.config.OD * self.z_m
            -2 * (self.x_m**2 + self.y_m**2) / self.config.w**2
        ).unsqueeze(-1)

        # Collect all eight internal components at each selected spatial node.
        spatial_components = []
        for idx in spatial_indices:
            start = int(idx) * 8
            spatial_components.extend(range(start, start + 8))
        spatial_components = torch.tensor(spatial_components, device=self.device)

        V_P_components = V[spatial_components, :].reshape(self.M, 8, self.r)
        rho_bar_P_components = rho_bar[spatial_components].reshape(self.M, 8)

        # calF and f satisfy
        # [A_OP (V*alpha+rho_bar)]_{q_s} = (calF*alpha)_s + f_s.
        A_op_V = torch.einsum('qd,mdr->mqr', self.config.A_op, V_P_components)
        A_op_rho_bar = torch.einsum(
            'qd,md->mq', self.config.A_op, rho_bar_P_components
        )
        selected_rows = torch.arange(self.M, device=self.device)
        self.calF = A_op_V[selected_rows, self.comp_indices, :].contiguous()
        self.f = A_op_rho_bar[
            selected_rows, self.comp_indices
        ].unsqueeze(-1).contiguous()

        # Collect the reduced basis and mean state along every selected
        # upstream axial ray. Duplicate rays are intentionally retained to
        # preserve the one-selected-entry/one-row convention used by DEIM.
        base = torch.arange(self.N, device=self.device)
        line_base_indices = (
            self.i_idx[:, None] * self.N * self.N +
            self.j_idx[:, None] * self.N +
            base[None, :]
        )
        line_spatial = line_base_indices.reshape(-1)
        line_integral_components = (
            line_spatial[:, None] * 8 +
            torch.arange(8, device=self.device)[None, :]
        ).reshape(-1)
        V_lines = V[line_integral_components, :].reshape(
            self.M, self.N, 8, self.r
        )
        rho_bar_lines = rho_bar[line_integral_components].reshape(
            self.M, self.N, 8
        )

        S_V_lines = torch.einsum('d,mkdr->mkr', self.config.S_z, V_lines)
        S_rho_bar_lines = torch.einsum(
            'd,mkd->mk', self.config.S_z, rho_bar_lines
        )

        # Boundary augmentation used by the manuscript integral.
        # The POD basis vanishes at prescribed Dirichlet nodes, whereas the
        # affine mean is completed with the physical boundary state rho=1/8.
        S_V_full = torch.zeros(self.M, self.N+2, self.r, device=self.device)
        S_V_full[:, 1:-1, :] = S_V_lines

        rho_bc = torch.full((8,), self.config.bc_value, device=self.device)
        S_rho_bc = torch.dot(self.config.S_z, rho_bc)
        S_rho_bar_full = torch.full(
            (self.M, self.N+2), S_rho_bc.item(), device=self.device
        )
        S_rho_bar_full[:, 1:-1] = S_rho_bar_lines

        # Each selected interior ray uses the cell widths from the full grid.
        dz_selected = self.config.dz[
            self.i_idx+1, self.j_idx+1, :, 0
        ].to(self.device)

        trapezoid_V = 0.5 * (
            S_V_full[:, :-1, :] + S_V_full[:, 1:, :]
        ) * dz_selected.unsqueeze(-1)
        trapezoid_rho_bar = 0.5 * (
            S_rho_bar_full[:, :-1] + S_rho_bar_full[:, 1:]
        ) * dz_selected

        integral_V = torch.zeros_like(S_V_full)
        integral_rho_bar = torch.zeros_like(S_rho_bar_full)
        integral_V[:, 1:, :] = torch.cumsum(trapezoid_V, dim=1)
        integral_rho_bar[:, 1:] = torch.cumsum(trapezoid_rho_bar, dim=1)

        # calE*alpha+e is the integral from z=0 to the selected physical node.
        selected_k_full = self.k_idx + 1
        self.calE = integral_V[selected_rows, selected_k_full, :].contiguous()
        self.e = integral_rho_bar[
            selected_rows, selected_k_full
        ].unsqueeze(-1).contiguous()

    def rom_rhs_Gop_DEIM(self, alpha):

        # I(alpha) = calE*alpha+e
        I_alpha = self.calE @ alpha + self.e

        # [A_OP*rho_r]_P = calF*alpha+f
        local_action_P = self.calF @ alpha + self.f

        # f_OP,P(alpha) = R0*xi_P*exp[2*OD*I(alpha)]*(calF*alpha+f)
        f_OP_P = (
            self.config.R0
            * self.xi_P
            * torch.exp(2 * self.config.OD * I_alpha)
            * local_action_P
        )
        self.rhs_Gop_r = self.GopTmp @ f_OP_P

        return self.rhs_Gop_r
