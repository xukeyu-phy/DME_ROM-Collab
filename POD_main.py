
import torch
from pathlib import Path
from POD_DEIM import POD_DEIM

device = torch.device("cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)

current_dir = Path(__file__).parent
dataset_dir = current_dir / "DME_dataset"
outdata_dir = current_dir / "Out_data"
fig_dir = current_dir / "Fig"
fig_dir.mkdir(exist_ok=True)

pod = POD_DEIM(dataset_dir, outdata_dir, fig_dir, device, dtype)
pod._main_line(recompute=True)