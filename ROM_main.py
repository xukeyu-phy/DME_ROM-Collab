__all__ = ['device', 'dtype']

import json
import torch
from pathlib import Path
from config import Config
from ROM_solver import ROMSolver
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'



current_dir = Path(__file__).parent
outdata_dir = current_dir / "Out_data"
fig_dir = current_dir / "Fig"

device = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)
torch.set_num_threads(1)

def main():
    config = Config(device, dtype)
    config._create_non_uniform_grid(outdata_dir)

    Phi_h = torch.load(outdata_dir / "POD_Phi.pt", weights_only=True).to(device)
    Phi_F = torch.load(outdata_dir / "DEIM_Phi.pt", weights_only=True).to(device)
    P_F = torch.load(outdata_dir / "DEIM_indices.pt", weights_only=True).to(device)
    POD_mean = torch.load(outdata_dir / "POD_mean.pt", weights_only=True).to(device)

    r = 1
    rf = 1
    Phi_r = Phi_h[:, :r]
    Phi_f = Phi_F[:, :rf]
    P_f = P_F[:rf]

    param_filename = current_dir / 'parameters.json'
    with open(param_filename, 'r') as f:
        phy_dict = json.load(f)
    romsolver = ROMSolver(r, rf, Phi_r, Phi_f, P_f, POD_mean, phy_dict, device, dtype, outdata_dir)
    
    dt = config._get_time_step(phy_dict['D'])
    result = romsolver._main_line(dt)
    
    Nh = Phi_r.shape[0]
    N3 = Nh // 8
    N = round(N3 ** (1/3))
    
    recon_rho_in = Phi_r @ result.cpu() + POD_mean
    recon_rho_in = recon_rho_in.reshape(N, N, N, 8)
    recon_rho = torch.zeros([N+2, N+2, N+2, 8])
    recon_rho[:, :, :, :] = 0.125
    recon_rho[1:-1, 1:-1, 1:-1, :] = recon_rho_in
    torch.save(recon_rho, outdata_dir / 'ROM_result.pt')

    S_z = torch.tensor([0.5, 0.25, 0, -0.25, -0.5, 0.25, 0, -0.25])
    P = 2 * torch.einsum('i, klmi -> klm', S_z, recon_rho)
    P = P.cpu()
    torch.save(P, outdata_dir/'ROM_P.pt')  


if __name__ == "__main__":
    main()

