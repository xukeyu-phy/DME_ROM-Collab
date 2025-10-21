import torch

class Config:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype       
        torch.set_default_dtype(dtype)
                
        # Numerical Parameters  
        self.T_final = 100.0       
        self.cfl = 0.1              
        self.convergence_tol = 1e-8  # Convergence tolerance
        self.ghostcell = 0           # Number of ghost cells
        
        # non-uniform grids
        self.n1 = 20           
        self.n2 = 60
        self.n3 = 20            
        self.grid_segments = [
            (0.00, 0.20),    # First segment
            (0.20, 0.80),    # Second segment
            (0.80, 1.00)]    # Third segment        
        
        # Equal rational grid cells for HF simulation
        self.N = 100         

        # Boundary Conditions
        self.bc_type = 'dirichlet'  # Options: 'dirichlet'
        self.bc_value = 0.125       # Boundary value for Dirichlet BC
        
        # Initial Conditions
        self.initial_condition_type = 'uniform' # Options: 'uniform'
        self.initial_value = 0.125              # Initial value for uniform condition
    
    def _get_time_step(self, D):
        min_dx = torch.min(self.dx)  
        min_dy = torch.min(self.dy)  
        min_dz = torch.min(self.dz)
        dd_min = min(min_dx, min_dy, min_dz)
        dt = self.cfl * dd_min **2 / (6 * D)
        # dt = self.cfl * dd_min
        return dt.clone().detach().to(device=self.device, dtype=self.dtype)
    
    def _create_non_uniform_grid(self, data_dir):
        data_dir.mkdir(exist_ok=True)
        segments = self.grid_segments
        n1, n2, n3 = self.n1, self.n2, self.n3
        gc = self.ghostcell

        coords_1 = torch.linspace(segments[0][0], segments[0][1], n1+1, 
                                 device=self.device, dtype=self.dtype)
        coords_2 = torch.linspace(segments[1][0], segments[1][1], n2+1, 
                                 device=self.device, dtype=self.dtype)[1:]  
        coords_3 = torch.linspace(segments[2][0], segments[2][1], n3+1, 
                                 device=self.device, dtype=self.dtype)[1:]  
        coords = torch.cat([coords_1, coords_2, coords_3])

        if gc > 0:
            lower_spacing = coords[1] - coords[0]
            lower_ghost = torch.linspace(
                coords[0] - gc * lower_spacing, 
                coords[0] - lower_spacing, 
                gc,
                device=self.device, dtype=self.dtype)            
            upper_spacing = coords[-1] - coords[-2]
            upper_ghost = torch.linspace(
                coords[-1] + upper_spacing, 
                coords[-1] + gc * upper_spacing, 
                gc,
                device=self.device, dtype=self.dtype)
            coords = torch.cat([lower_ghost, coords, upper_ghost])

        x_coords = coords - 0.5
        y_coords = coords - 0.5
        z_coords = coords.clone()
        
        grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')    
        coord_tensor = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        
        self.dx = grid_x[1:, :, :] - grid_x[:-1, :, :]
        self.dy = grid_y[:, 1:, :] - grid_y[:, :-1, :]
        self.dz = grid_z[:, :, 1:] - grid_z[:, :, :-1]    

        if len(self.dx.shape) == 3:
            self.dx = self.dx.unsqueeze(-1)
        if len(self.dy.shape) == 3:
            self.dy = self.dy.unsqueeze(-1)
        if len(self.dz.shape) == 3:
            self.dz = self.dz.unsqueeze(-1)

        grid = coord_tensor
        self.Nx, self.Ny, self.Nz = grid.shape[0], grid.shape[1], grid.shape[2]

        Mesh = {
        'grid': coord_tensor.cpu(),
        'grid_x': grid_x.cpu(),
        'grid_y': grid_y.cpu(),
        'grid_z': grid_z.cpu(), 
        'dx': self.dx.cpu(),
        'dy': self.dy.cpu(),
        'dz': self.dz.cpu()}
        torch.save(Mesh, data_dir/'Mesh.pt')
        
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z

        return coord_tensor, self.grid_x, self.grid_y, self.grid_z,self.dx, self.dy, self.dz
    

    def _create_equal_ratio_grid(self, data_dir, phy_dict):
        self._setup_phy_ps(phy_dict)
        lam = torch.sqrt(2 * self.D / (self.R0 * torch.exp(-self.OD)))
        self.h1 = float(0.05 * lam)
        if self.N % 2 != 0:  
            print(f"Note: self.N={self.N} is odd, automatically modified to {self.N+1}")
            self.N += 1
        Np, n = self.N + 1, self.N // 2

        # === Solve alpha ===
        def geom_sum(a):  
            return self.h1 * n if abs(a - 1) < 1e-14 else self.h1 * (1 - a**n) / (1 - a) 

        f = lambda a: geom_sum(a) - 0.5
        left, right = 1.0, 2.0
        for _ in range(self.N): 
            if f(left) * f(right) < 0: break
            right *= 2
        for _ in range(self.N): 
            mid = 0.5 * (left + right)
            if f(left) * f(mid) <= 0: right = mid
            else: left = mid
            if abs(right - left) < 1e-10: break
        alpha = 0.5 * (left + right)
        print(f"alpha = {alpha:.6f}")

        # === Construct the physical grid ===
        x_left = torch.zeros(n + 1, device=self.device)
        for i in range(1, n + 1):
            x_left[i] = x_left[i - 1] + self.h1 * (alpha ** (i - 1))
        x_left[-1] = 0.5
        x_physical = torch.cat([x_left[:-1], 1 - x_left.flip(0)])
        xi_computational = torch.linspace(0, 1, Np, device=self.device)

        coords = x_physical - 0.5
        grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, x_physical, indexing='ij')
        coord_tensor = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        
        self.dx = (grid_x[1:] - grid_x[:-1]).unsqueeze(-1)
        self.dy = (grid_y[:, 1:] - grid_y[:, :-1]).unsqueeze(-1)
        self.dz = (grid_z[:, :, 1:] - grid_z[:, :, :-1]).unsqueeze(-1)

        data_dir.mkdir(parents=True, exist_ok=True)
        Mesh = {
            'grid': coord_tensor.cpu(), 
            'grid_x': grid_x.cpu(), 
            'grid_y': grid_y.cpu(),
            'grid_z': grid_z.cpu(), 
            'dx': self.dx.cpu(), 
            'dy': self.dy.cpu(), 
            'dz': self.dz.cpu()
        }
        torch.save(Mesh, data_dir / 'Mesh.pt')
        self.grid_x, self.grid_y, self.grid_z = grid_x, grid_y, grid_z
        self.Nx, self.Ny, self.Nz = grid_x.shape
        
        return coord_tensor, grid_x, grid_y, grid_z, self.dx, self.dy, self.dz



    def _setup_phy_ps(self, param):
        normal_param = param
        self.D = torch.tensor(normal_param['D'], device=self.device, dtype=torch.float64)
        self.w = torch.tensor(normal_param['w'], device=self.device)
        self.eta = torch.tensor(normal_param['eta'], device=self.device)
        self.R0 = torch.tensor(normal_param['R0'], device=self.device)
        self.fD = torch.tensor(normal_param['fD'], device=self.device)
        self.OD = torch.tensor(normal_param['OD'], device=self.device)
        self.Qa = torch.tensor(normal_param['Qa'], device=self.device)
        self.Qb = torch.tensor(normal_param['Qb'], device=self.device)

    def _setup_matrices(self, Qa=1.0, Qb=1.0):
        self.Qa = Qa
        self.Qb = Qb
        self.A_SD = 0.0625 * torch.tensor([
            [8, -2, 0, 0, 0, 0, 0, -6],
            [-2, 11, -3, 0, 0, 0, -3, -3],
            [0, -3, 12, -3, 0, -1, -4, -1],
            [0, 0, -3, 11, -2, -3, -3, 0],
            [0, 0, 0, -2, 8, -6, 0, 0],
            [0, 0, -1, -3, -6, 11, -1, 0],
            [0, -3, -4, -3, 0, -1, 12, -1],
            [-6, -3, -1, 0, 0, 0, -1, 11]
        ], device=self.device, dtype=torch.float64)
        
        self.A_FD = torch.tensor([
            [2, -2, 0, 0, 0, 0, 0, 0],
            [-2, 5, -3, 0, 0, 0, 0, 0],
            [0, -3, 6, -3, 0, 0, 0, 0],
            [0, 0, -3, 5, -2, 0, 0, 0],
            [0, 0, 0, -2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, 0],
            [0, 0, 0, 0, 0, -1, 2, -1],
            [0, 0, 0, 0, 0, 0, -1, 1]
        ], device=self.device, dtype=torch.float64)

        self.A_op = torch.tensor([
            [0, -0.25, 0, 0, 0, 0, 0, -0.75],
            [0, 0.4375, -0.375, 0, 0, 0, -0.375, -0.1875],
            [0, 0, 0.75, -0.375, 0, -0.125, -0.25, 0],
            [0, 0, 0, 0.9375, -0.25, -0.1875, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, -0.1875, -0.75, 0.4375, 0, 0],
            [0, 0, -0.25, -0.375, 0, -0.125, 0.75, 0],
            [0, -0.1875, -0.125, 0, 0, 0, -0.125, 0.9375]
        ], device=self.device, dtype=torch.float64)

        self.Q_ab = torch.tensor([
            [self.Qa, 0, 0, 0, 0, 0, 0, 0],
            [0, self.Qa, 0, 0, 0, 0, 0, 0],
            [0, 0, self.Qa, 0, 0, 0, 0, 0],
            [0, 0, 0, self.Qa, 0, 0, 0, 0],
            [0, 0, 0, 0, self.Qa, 0, 0, 0],
            [0, 0, 0, 0, 0, self.Qb, 0, 0],
            [0, 0, 0, 0, 0, 0, self.Qb, 0],
            [0, 0, 0, 0, 0, 0, 0, self.Qb]
        ], device=self.device, dtype=torch.float64)

        self.A_op = self.A_op @ self.Q_ab
        
        self.A_SE = 0.125 * torch.tensor([
            [8, 2, 0, 0, 0, 0, 0, 6],
            [-2, 4, 3, 0, 0, 0, 3, 0],
            [0, -3, 0, 3, 0, 1, 0, -1],
            [0, 0, -3, -4, 2, 0, -3, 0],
            [0, 0, 0, -2, -8, -6, 0, 0],
            [0, 0, -1, 0, 6, 4, -1, 0],
            [0, -3, 0, 3, 0, 1, 0, -1],
            [-6, 0, 1, 0, 0, 0, 1, -4]
        ], device=self.device, dtype=torch.float64)
        
        self.S_z = torch.tensor(
            [0.5, 0.25, 0, -0.25, -0.5, 0.25, 0, -0.25],
            device=self.device, dtype=torch.float64)
        
    def _setup_coefficient(self):
        # x direction
        dx_im1 = self.dx[:-1, 1:-1, 1:-1, :]  # dx_{i-1/2}
        dx_ip1 = self.dx[1:, 1:-1, 1:-1, :]   # dx_{i+1/2}        
        
        self.alpha_x = 2.0 / (dx_im1 * (dx_im1 + dx_ip1))    # Calculate weight coefficient 
        self.beta_x = -2.0 / (dx_im1 * dx_ip1)
        self.gamma_x = 2.0 / (dx_ip1 * (dx_im1 + dx_ip1))

        # y direction
        dy_jm1 = self.dy[1:-1, :-1, 1:-1, :]  # dy_{j-1/2}
        dy_jp1 = self.dy[1:-1, 1:, 1:-1, :]   # dy_{j+1/2}
        
        self.alpha_y = 2.0 / (dy_jm1 * (dy_jm1 + dy_jp1))
        self.beta_y = -2.0 / (dy_jm1 * dy_jp1)
        self.gamma_y = 2.0 / (dy_jp1 * (dy_jm1 + dy_jp1))

        # z direction
        dz_km1 = self.dz[1:-1, 1:-1, :-1, :]  # dz_{k-1/2}
        dz_kp1 = self.dz[1:-1, 1:-1, 1:, :]   # dz_{k+1/2}
        
        self.alpha_z = 2.0 / (dz_km1 * (dz_km1 + dz_kp1))
        self.beta_z = -2.0 / (dz_km1 * dz_kp1)
        self.gamma_z = 2.0 / (dz_kp1 * (dz_km1 + dz_kp1))   
    



    def __repr__(self):
        return f"""Numerical Config(
                    grid=(n1:{self.n1}, n2:{self.n2}, n3:{self.n3}, N:{self.N}), segments={self.grid_segments},
                    T_final={self.T_final}, ghostcell={self.ghostcell}
                )"""

