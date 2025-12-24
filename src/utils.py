import os
import json
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(device: str = "auto"):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_state_dict_safely(model: torch.nn.Module, state_dict: dict):
    # Allows loading checkpoints even if wrapped/unwrapped
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {"missing": missing, "unexpected": unexpected}

def model_to_cpu_state(model: torch.nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def add_dp_noise_to_state(delta_state: dict, sigma: float, clip_norm: float, seed: int = 0):
    """Simple DP-style update: clip L2 norm of the full update vector, then add Gaussian noise.
    This is a **didactic** implementation; for real DP, use a proper accountant.
    """
    if sigma <= 0:
        return delta_state

    # Flatten to compute norm
    torch.manual_seed(seed)
    flats = []
    for v in delta_state.values():
        flats.append(v.view(-1).float())
    vec = torch.cat(flats)
    norm = torch.norm(vec, p=2).item()
    scale = 1.0
    if clip_norm > 0 and norm > clip_norm:
        scale = clip_norm / (norm + 1e-12)

    noisy = {}
    for k, v in delta_state.items():
        vv = (v.float() * scale)
        noise = torch.randn_like(vv) * sigma
        noisy[k] = (vv + noise).type_as(v)
    return noisy
