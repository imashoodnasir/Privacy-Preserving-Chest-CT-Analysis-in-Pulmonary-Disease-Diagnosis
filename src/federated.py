import copy
import numpy as np
import torch

from .utils import model_to_cpu_state, add_dp_noise_to_state

def state_diff(new_state: dict, old_state: dict):
    return {k: (new_state[k] - old_state[k]) for k in new_state.keys()}

def apply_state_update(base_state: dict, delta_state: dict, weight: float = 1.0):
    out = {}
    for k in base_state.keys():
        out[k] = base_state[k] + delta_state[k] * weight
    return out

def fedavg_aggregate(global_state: dict, client_states: list, client_sizes: list,
                     dp_sigma: float = 0.0, dp_clip: float = 0.0, seed: int = 0):
    """FedAvg aggregation over *model deltas*.
    client_states are full states AFTER local training. We aggregate deltas relative to global_state.

    dp_sigma, dp_clip implement a simple didactic DP update (noise added to deltas).
    """
    total = float(sum(client_sizes))
    agg_delta = {k: torch.zeros_like(v) for k, v in global_state.items()}
    for i, (st, n) in enumerate(zip(client_states, client_sizes)):
        delta = state_diff(st, global_state)
        delta = add_dp_noise_to_state(delta, sigma=dp_sigma, clip_norm=dp_clip, seed=seed + i)
        w = float(n) / total
        for k in agg_delta.keys():
            agg_delta[k] += delta[k] * w
    new_global = apply_state_update(global_state, agg_delta, weight=1.0)
    return new_global

def select_clients(client_paths: list, clients_per_round: int, seed: int):
    rng = np.random.default_rng(seed)
    if clients_per_round >= len(client_paths):
        return client_paths
    idx = rng.choice(len(client_paths), size=clients_per_round, replace=False)
    return [client_paths[i] for i in idx]
