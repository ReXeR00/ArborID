import random 
import numpy as np 
import torch
from omegaconf import DictConfig

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def setup_amp(amp_dtype: str, device: torch.device):
    use_amp = str(amp_dtype).lower() == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_t = torch.float16 if use_amp else torch.float32
    return use_amp, scaler, amp_t   