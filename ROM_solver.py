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
        self.P_f = P_f
        self.Nh = Phi_r.shape[0]
        self.N3 = self.Nh // 8
        self.N = round(self.N3 ** (1/3))

        self.config._create_non_uniform_grid(outdata_dir)
        self.config._setup_coefficient()        

        self._init_phy_ps(phy_ps)
        self.POD_mean = POD_mean
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
            rho_r_init = self.Phi_r.T @ (rho_init - self.POD_mean.reshape(-1,1)) 

            t = 0.0
            t_iter = 0
            rho_r_n = rho_r_init.clone()   
            while t < self.config.T_final:
                if (t + dt) >= self.config.T_final:
                    dt = self.config.T_final - t
                t += dt
                t_iter += 1 

                if t_iter % 200 == 1:                                
                    rho_r_n0 = rho_r_n.clone()               
                    rho_r_n = self.runge_kutta_2_step(rho_r_n, dt, self.config.ghostcell, self.config.bc_type, 'update')
                    drho = torch.abs(rho_r_n - rho_r_n0)

                    l2_drho = torch.norm(drho, p=2) 
                    print(f"Iter: {t_iter}, Time: {t:.6f}, l2_drho = {l2_drho:.4e}")                    
                    if l2_drho < self.config.convergence_tol:
                        print(f'Convergence: l2_drho = {l2_drho:.6e}')
                        break
                else:
                    rho_r_n = self.runge_kutta_2_step(rho_r_n, dt, self.config.ghostcell, self.config.bc_type, None)

            iter_count += 1        

        end_time = time.time()
        runtime =  end_time - start_time
        print(f"Runtime: {runtime:.3f}s, 1 case time: {runtime/iter_count:3f}")
        # self.print_deim_timing()
        return rho_r_n

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
        self.C_NQE_rho_r = C_NQE

    def rom_rhs_diffusion(self, rho_r):        
        rho_r = self.nabla_r @ rho_r
        return rho_r

    def rom_rhs_G0(self, rho_r):
        rhs_G0_r = self.G0_r @ rho_r        
        return rhs_G0_r  

    def rom_rhs_GNL_NQE(self, rho_r):
        rho_r_flat = rho_r.flatten()
        rho_outer = torch.outer(rho_r_flat, rho_r_flat)
        rho_outer_flat = rho_outer.flatten().unsqueeze(0) 
        A_NQE_rho_r = self.A_NQE @ rho_outer_flat.T
        B_NQE_rho_r = self.B_NQE @ rho_r

        rhs_GNL_r =  -self.config.eta * (A_NQE_rho_r + B_NQE_rho_r + self.C_NQE_rho_r)
        return rhs_GNL_r

    def rhs(self, rho, update):
        if update == 'update':
            rhs_vib =  - self.rom_rhs_G0(rho) - self.rom_rhs_Gop_DEIM(rho) - self.rom_rhs_GNL_NQE(rho) + self.rom_rhs_diffusion(rho) 
        else:
            rhs_vib =  - self.rom_rhs_G0(rho) - self.rhs_Gop_r - self.rom_rhs_GNL_NQE(rho) + self.rom_rhs_diffusion(rho) 
        rhs = rhs_vib + self.rhs_mean

        return rhs


    def runge_kutta_2_step(self, rho_n, dt, ghostcell=None, bc_type=None, update=None):
        k1 = self.rhs(rho_n, update)
        rho_1 = rho_n + 1.0 * dt * k1

        k2 = self.rhs(rho_1, update)        
        rho_new = rho_n + (dt/2.0) * (k1 + k2)
        
        return rho_new
    

    def rom_DEIM_preperform(self):
        x, y, z = self.config.grid_x, self.config.grid_y, self.config.grid_z
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        z = z.unsqueeze(-1)
        x = x.to(self.device)
        y = y.to(self.device)
        self.config.w = self.config.w.to(self.device)
        
        exp_pre = torch.exp(-self.config.OD * z) * torch.exp(-2 * (x**2 + y**2)/self.config.w**2)
        exp_pre = exp_pre[1:-1, 1:-1, 1:-1]
        
        Phi_Gop_deim_P = self.Phi_Gop_deim[self.P_f, :]
        self.GopTmp = self.Phi_r.T @ self.Phi_Gop_deim @ torch.linalg.inv(Phi_Gop_deim_P)

        spatial_indices = torch.div(self.P_f, 8, rounding_mode='floor')
        self.comp_indices = self.P_f % 8

        self.M = spatial_indices.numel()

        self.k_idx = (spatial_indices % self.N).to(self.device)
        self.j_idx = ((spatial_indices // self.N) % self.N).to(self.device)
        self.i_idx = (spatial_indices // (self.N * self.N)).to(self.device)        

        spatial_components = []
        for idx in spatial_indices:
            start = int(idx) * 8
            spatial_components.extend(range(start, start + 8))
        spatial_components = torch.tensor(spatial_components, device=self.device)
        self.Phi_r_P_components = self.Phi_r[spatial_components, :].contiguous()
        self.mean_P_components = self.POD_mean[spatial_components].reshape(-1, 1)
        
        exp_pre_flat = exp_pre.reshape(-1)
        self.exp_pre = exp_pre_flat[spatial_indices].unsqueeze(-1)
        
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
        self.Phi_r_integral_components = self.Phi_r[line_integral_components, :].contiguous()
        self.mean_integral_components = self.POD_mean[line_integral_components].reshape(-1, 1)        

        self.rho_integral_components_full = torch.zeros(self.M, self.N+2, 8, device=self.device)
        self.rho_integral_components_full[:, :, :] = 0.125

        self.integral_component = torch.zeros_like(self.rho_integral_components_full)
        
        dz_intergral = self.config.dz[0, 0, 0]
        k_positions = torch.arange(self.N + 1, device=self.device)
        k_idx_expanded = self.k_idx.unsqueeze(1)  # [M, 1]

        self.integral_weights = (k_positions.unsqueeze(0) <= k_idx_expanded).float() * dz_intergral[0]
        self.integral_weights = self.integral_weights.unsqueeze(-1)

    def rom_rhs_Gop_DEIM(self, rho_r):

        rho_lines = self.Phi_r_integral_components @ rho_r + self.mean_integral_components
        rho_lines = rho_lines.reshape(self.M, self.N, 8)
        self.rho_integral_components_full[:, 1:-1, :] = rho_lines
        
        rho_left = self.rho_integral_components_full[:, :-1, :]  
        rho_right = self.rho_integral_components_full[:, 1:, :]  
        
        trapezoid_areas = 0.5 * (rho_left + rho_right) * self.integral_weights       

        integral_component = torch.sum(trapezoid_areas, dim=1)
        integral = torch.sum(self.config.S_z[None, :] * integral_component, dim=1)
        G_OP = torch.exp(2 * self.config.OD * integral)
        
        rho_P_components = self.Phi_r_P_components @ rho_r + self.mean_P_components
        rho_P_components = rho_P_components.reshape(self.M, 8)
        
        A_op_rho = torch.mm(rho_P_components, self.config.A_op.T)
        A_op_rho = torch.gather(A_op_rho, 1, self.comp_indices.unsqueeze(-1))

        rhs_DEIM_hp_selected = self.config.R0 * A_op_rho * G_OP.unsqueeze(-1) * self.exp_pre
        self.rhs_Gop_r = self.GopTmp @ rhs_DEIM_hp_selected
        
        return self.rhs_Gop_r