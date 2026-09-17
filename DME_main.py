__all__ = ['device', 'dtype']

import json
import torch
from pathlib import Path
from config import Config
from DME_solver import DMESolver

current_dir = Path(__file__).parent
outdata_dir = current_dir / "Out_data"
outdata_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)
grid_type = 'uniform'  # Optional: 'uniform' or 'ratio'

def main():
    param_filename = current_dir / 'parameters.json'
    with open(param_filename, 'r') as f:
        phy_dict = json.load(f)
        config = Config(device, dtype)
    
    if grid_type == 'ratio':    
        config._create_equal_ratio_grid(outdata_dir, phy_dict)
    else:
        config._create_non_uniform_grid(outdata_dir)
    dmesolver = DMESolver(device, dtype, phy_dict, grid_type, outdata_dir)
    dmesolver._init_phy_ps(phy_dict)
    dt = config._get_time_step(phy_dict)

    result, xi, xiz = dmesolver._main_line(dt)
    torch.save(result, outdata_dir / f'HF_result.pt')
    torch.save(xi, outdata_dir / f'xi.pt')
    torch.save(xiz, outdata_dir / f'xiz.pt')


if __name__ == "__main__":
    main()
