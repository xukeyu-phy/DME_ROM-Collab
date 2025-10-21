import torch
import time
from config import Config


class DMESolver:
    def __init__(self, device, dtype, phy_ps, grid_type, xi_method, data_dir):
        self.device = device
        self.config = Config(device, dtype)
        torch.set_default_dtype(dtype)
        
        if grid_type == 'ratio':    
            self.config._create_equal_ratio_grid(data_dir, phy_ps)
        elif grid_type == 'uniform':
            self.config._create_non_uniform_grid(data_dir)
        else:
            raise ValueError(f"Unsupported grid type: {grid_type}")
        self.xi_method = xi_method
        self._init_phy_ps(phy_ps)
        self.config._setup_coefficient()        
        self.Nx = self.config.grid_x.shape[0]
        self.Ny = self.config.grid_y.shape[1]
        self.Nz = self.config.grid_z.shape[2]
        

    def _init_phy_ps(self, phy_ps):
        self.config._setup_phy_ps(phy_ps)
        self.config._setup_matrices(self.config.Qa, self.config.Qb)
        self._update_pump_distribution(None, 'init')


    def _main_line(self, dt):
        start_time = time.time()
        rho_init = torch.zeros((self.config.Nx, self.config.Ny, self.config.Nz, 8), device=self.device)
        rho_init = self._setup_initial_condition(rho_init)
        rho_init = self._setup_boundary_conditions(rho_init, self.config.ghostcell, self.config.bc_type, self.config.bc_value)

        t = 0.0
        t_iter = 0
        rho_n = rho_init.clone()        

        while t < self.config.T_final:
            if (t + dt) >= self.config.T_final:
                dt = self.config.T_final - t
            t += dt
            t_iter += 1                   
            if t_iter % 200 == 1:                   
                rho_n0 = rho_n.clone()               
                rho_n = self.runge_kutta_2_step(rho_n, dt, self.config.ghostcell, self.config.bc_type)
                drho = torch.abs(rho_n - rho_n0)
                l2_drho = torch.norm(drho, p=2)

                xiz_n0 = self.xi[self.Nx//2, self.Ny//2, :, :]
                self._update_pump_distribution(rho_n, self.xi_method)        
                xiz_n = self.xi[self.Nx//2, self.Ny//2, :, :]
                dxiz = torch.abs(xiz_n - xiz_n0)
                l2_dxiz = torch.norm(dxiz, p=2)

                print(f"Iter: {t_iter}, Time: {t:.6f}, l2_drho = {l2_drho:.4e}, l2_dxiz = {l2_dxiz:.4e} ")
                if l2_drho < self.config.convergence_tol and l2_dxiz < self.config.convergence_tol :
                    print(f'Convergence: l2_drho = {l2_drho:.6e}, l2_dxiz = {l2_dxiz:.4e}')
                    break
            else:
                rho_n = self.runge_kutta_2_step(rho_n, dt, self.config.ghostcell, self.config.bc_type)
            
        end_time = time.time()
        runtime =  end_time - start_time
        print(f"Runtime: {runtime:.3f}s")    
        return rho_n, self.xi


    def _update_pump_distribution(self, rho, method):
        x, y, z = self.config.grid_x, self.config.grid_y, self.config.grid_z
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        z = z.unsqueeze(-1)
        x = x.to(self.device)
        y = y.to(self.device)
        self.config.w = self.config.w.to(self.device)
        xi_xy = torch.exp(-2*(x**2 + y**2)/(self.config.w**2))
        xi_z = torch.ones_like(z, device=self.device)

        if method == 'approx':            
            xi_z = torch.exp(-self.config.OD * z)

        elif method == 'init':
            xi_z = torch.ones_like(z, device=self.device)

        elif method == 'Richardson':
            dz = self.config.dz
            Srho = torch.einsum('i,klmi->klm', self.config.S_z, rho)
            Srho = Srho.unsqueeze(-1)
            xi_z[:, :, 0, :] = 1.0
            xi_z_h = xi_z[:, :, 0, :] / (1 + self.config.OD * dz[:, :, 0] * (1 - 2 * Srho[:, :, 1, :]))
            Srho_half = ( Srho[:, :, 0, :] + Srho[:, :, 1, :] ) *0.5
            xi_z_half = xi_z[:, :, 0, :] /  (1 + self.config.OD * dz[:, :, 0] * 0.5 * (1 - 2 * Srho_half))
            xi_z_h2 = xi_z_half /  (1 + self.config.OD * dz[:, :, 0] * 0.5 * (1 - 2 * Srho[:, :, 1, :]))
            xi_z[:, :, 1, :] = 2 * xi_z_h2 - xi_z_h
            for ii in range(2, self.Nz):
                xi_z[:, :, ii, :] = (4 * xi_z[:, :, ii-1, :] - xi_z[:, :, ii-2, :]) / (3 + 2 * self.config.OD * dz[:, :, ii-1] * (1 - 2 * Srho[:, :, ii, :]))        

        elif method == 'BDF2':
            dz = self.config.dz
            Srho = torch.einsum('i,klmi->klm', self.config.S_z, rho)
            Srho = Srho.unsqueeze(-1)
            xi_z[:, :, 0, :] = 1.0
            xi_z[:, :, 1, :] = xi_z[:, :, 0, :] / (1 + self.config.OD * dz[:, :, 0] * (1 - 2 * Srho[:, :, 1, :]))            
            for ii in range(2, self.Nz):
                xi_z[:, :, ii, :] = (4 * xi_z[:, :, ii-1, :] - xi_z[:, :, ii-2, :]) / (3 + 2 * self.config.OD * dz[:, :, ii-1] * (1 - 2 * Srho[:, :, ii, :]))        

        elif method == 'RK2':
            dz = self.config.dz
            Srho = torch.einsum('i,klmi->klm', self.config.S_z, rho)
            Srho = Srho.unsqueeze(-1)
            xi_z[:, :, 0, :] = 1.0
            f1 = 0.5 * self.config.OD * dz[:, :, :] * (1 - 2 * Srho[:, :, :-1, :])
            f2 = 0.5 * self.config.OD * dz[:, :, :] * (1 - 2 * Srho[:, :, 1:, :])
            for ii in range(1, self.Nz):
                xi_z[:, :, ii, :] = xi_z[:, :, ii-1, :] * (1 - f1[:, :, ii-1, :]) / (1 + f2[:, :, ii-1, :])

        elif method == 'integral':
            dz = self.config.dz
            trapezoid_areas = 0.5 * (rho[:, :, :-1, :] + rho[:, :, 1:, :]) * dz
            integral_component = torch.zeros_like(rho)
            integral_component[:, :, 1:, :] = torch.cumsum(trapezoid_areas, dim=2)
            integral = torch.sum(self.config.S_z[None, None, None, :] * integral_component, dim=3)
            G_OP = torch.exp(2 * self.config.OD * integral) 
            exp_OD_z = torch.exp(-self.config.OD * self.config.grid_z)              

            xi_z = exp_OD_z * G_OP 
            xi_z = xi_z.unsqueeze(-1)

        else:
            print(f'Wrong in Xi_z update!')

        self.xi = xi_xy * xi_z


    def rhs(self, rho):
        rho = rho.to(self.device)
        
        diffusion = self.rhs_diffusion(rho)
        G0_term = self.rhs_G0(rho)
        Gop_term = self.rhs_Gop(rho)
        GNQE_term = self.rhs_GNQE(rho)
        
        spatial_rhs = torch.zeros_like(rho, device=self.device)
        interior = slice(1, -1)
        
        spatial_rhs[interior, interior, interior, :] = (
            diffusion[interior, interior, interior, :]
            - G0_term[interior, interior, interior, :] 
            - Gop_term[interior, interior, interior, :]
            - GNQE_term[interior, interior, interior, :])        
        return spatial_rhs

    
    def _setup_boundary_conditions(self, rho, gc, bc_type, bc_value):
        if bc_type == 'dirichlet':
            rho[0, :, :, :] = bc_value
            rho[-1, :, :, :] = bc_value
            rho[:, 0, :, :] = bc_value
            rho[:, -1, :, :] = bc_value
            rho[:, :, 0, :] = bc_value
            rho[:, :, -1, :] = bc_value
            
        elif bc_type == 'periodic':
            rho[:gc, :, :, :] = rho[-2*gc:-gc, :, :, :]  
            rho[-gc:, :, :, :] = rho[gc:2*gc, :, :, :]  

            rho[:, :gc, :, :] = rho[:, -2*gc:-gc, :, :] 
            rho[:, -gc:, :, :] = rho[:, gc:2*gc, :, :]
            
            rho[:, :, :gc, :] = rho[:, :, -2*gc:-gc, :]
            rho[:, :, -gc:, :] = rho[:, :, gc:2*gc, :]

        elif bc_type == 'neumann':
            pass
        else:
            raise ValueError(f"Unsupported boundary condition type: {bc_type}")
        
        return rho
    

    def _setup_initial_condition(self, rho):
        if self.config.initial_condition_type == 'uniform':
            rho[:, :, :, :] = self.config.initial_value
        elif self.config.initial_condition_type == 'analytical':
            pass        
        return rho
    

    def _setup_steady_state_solution(self):
        xo = 0.0
        yo = 0.0
        zo = 0.5
        xi_r0 = torch.exp(-(xo**2 + yo**2)/(2*self.config.w**2)) * torch.exp(-self.config.OD * zo)
        xi_r0 = xi_r0.clone().detach().to(dtype=torch.float64, device=self.device)
        R = self.config.R0 * xi_r0
        P = R / (R + 1)
        denominator = 8 * (P**2 + 1)
        
        rho_0 = torch.tensor([
            (P + 1)**4 / denominator,
            -(P - 1) * (P + 1)**3 / denominator,
            (P**2 - 1)**2 / denominator,
            -(P - 1)**3 * (P + 1) / denominator,
            (P - 1)**4 / denominator,
            -(P - 1)**3 * (P + 1) / denominator,
            (P**2 - 1)**2 / denominator,
            -(P - 1) * (P + 1)**3 / denominator
        ], device=self.device, dtype=self.dtype)        
        return rho_0
    
    
    def rhs_diffusion(self, rho):
        d2rho_dx2 = (self.config.alpha_x * rho[:-2, 1:-1, 1:-1, :] + 
                     self.config.beta_x * rho[1:-1, 1:-1, 1:-1, :] + 
                     self.config.gamma_x * rho[2:, 1:-1, 1:-1, :])
        
        d2rho_dy2 = (self.config.alpha_y * rho[1:-1, :-2, 1:-1, :] + 
                     self.config.beta_y * rho[1:-1, 1:-1, 1:-1, :] + 
                     self.config.gamma_y * rho[1:-1, 2:, 1:-1, :])
        
        d2rho_dz2 = (self.config.alpha_z * rho[1:-1, 1:-1, :-2, :] + 
                     self.config.beta_z * rho[1:-1, 1:-1, 1:-1, :] + 
                     self.config.gamma_z * rho[1:-1, 1:-1, 2:, :])
        nabla2 = rho.clone()
        nabla2[1:-1, 1:-1, 1:-1, :] = d2rho_dx2 + d2rho_dy2 + d2rho_dz2
        return self.config.D * nabla2
    
    def rhs_G0(self, rho):
        self.G0_matrix = (1 + self.config.eta) * self.config.A_SD + self.config.fD * self.config.A_FD
        rhs_G0 = torch.einsum('ij,klmj->klmi', self.G0_matrix, rho)
        return rhs_G0
    
    def rhs_Gop(self, rho):      
        A_op_rho = torch.einsum('ij,klmj->klmi', self.config.A_op, rho)
        rhs_Gop = self.config.R0 * self.xi * A_op_rho
        return rhs_Gop
    
    def rhs_GNQE(self, rho):
        S_dot_rho = torch.einsum('i,klmi->klm', self.config.S_z, rho)
        S_dot_rho_expanded = S_dot_rho.unsqueeze(-1)
        A_SE_rho = torch.einsum('ij,klmj->klmi', self.config.A_SE, rho)
        rhs_GNL = -self.config.eta * S_dot_rho_expanded * A_SE_rho
        return rhs_GNL
    

    def runge_kutta_2_step(self, rho_n, dt, ghostcell=None, bc_type=None):

        k1 = self.rhs(rho_n)
        rho_1 = rho_n + 1.0 * dt * k1
        
        k2 = self.rhs(rho_1)        
        rho_new = rho_n + (dt/2.0) * (k1 + k2)
        
        return rho_new

 