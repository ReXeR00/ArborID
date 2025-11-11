import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from data import get_loaders
from model import model as build_model




