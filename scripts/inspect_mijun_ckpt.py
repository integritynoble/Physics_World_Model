#!/usr/bin/env python3
"""Inspect recovered MiJUN checkpoint structure."""
import torch
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "reference", "cassi", "our_mamba_5stg_recovered.pth")

state = torch.load(CKPT, map_location='cpu', weights_only=False)
print(f"Top-level keys: {list(state.keys())}")
print(f"Epoch: {state.get('epoch')}")

# Inspect model dict
model_state = state.get('model', {})
if hasattr(model_state, 'keys'):
    keys = sorted(model_state.keys())
    print(f"\nModel state_dict: {len(keys)} keys")

    # Show unique prefixes
    prefixes = set()
    for k in keys:
        parts = k.split('.')
        if len(parts) >= 2:
            prefixes.add(f"{parts[0]}.{parts[1]}")
        else:
            prefixes.add(parts[0])
    print(f"Unique prefixes: {sorted(prefixes)}")

    # Show all keys with shapes
    total_params = 0
    for k in keys:
        v = model_state[k]
        if hasattr(v, 'shape'):
            total_params += v.numel()
            print(f"  {k}: {list(v.shape)} ({v.dtype})")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")

    print(f"\nTotal parameters: {total_params:,}")
else:
    print(f"model value type: {type(model_state)}")

# Check EMA too
ema_state = state.get('ema', {})
if hasattr(ema_state, 'keys'):
    print(f"\nEMA state_dict: {len(ema_state)} keys")
