from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")
    return checkpoint


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    elif checkpoint and all(isinstance(key, str) for key in checkpoint.keys()):
        state_dict = checkpoint
    else:
        raise ValueError("Could not find a model state dict in the checkpoint.")

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    return state_dict


def load_state_dict(path: str | Path, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = load_checkpoint(path, device=device)
    state_dict = extract_state_dict(checkpoint)
    return checkpoint, state_dict
