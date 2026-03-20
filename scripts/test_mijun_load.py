#!/usr/bin/env python3
"""Test loading MiJUN checkpoint into the architecture."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core', 'pwm_core', 'recon'))

import torch
from cassi_arch.MiJUN import MiJUN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "reference", "cassi", "our_mamba_5stg_recovered.pth")

# Build model
model = MiJUN(stage=5, n_bands=28, step=2)

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total:,}")

# Load checkpoint
state = torch.load(CKPT, map_location='cpu', weights_only=False)
ckpt_sd = state['model']
print(f"Checkpoint keys: {len(ckpt_sd)}")

# Compare keys
model_keys = set(model.state_dict().keys())
ckpt_keys = set(ckpt_sd.keys())

missing = model_keys - ckpt_keys
extra = ckpt_keys - model_keys

if missing:
    print(f"\n{len(missing)} keys in model but NOT in checkpoint:")
    for k in sorted(missing)[:20]:
        print(f"  MISSING: {k} -> {list(model.state_dict()[k].shape)}")

if extra:
    print(f"\n{len(extra)} keys in checkpoint but NOT in model:")
    for k in sorted(extra)[:20]:
        print(f"  EXTRA: {k} -> {list(ckpt_sd[k].shape)}")

# Check shape mismatches for common keys
common = model_keys & ckpt_keys
mismatches = []
for k in sorted(common):
    ms = list(model.state_dict()[k].shape)
    cs = list(ckpt_sd[k].shape)
    if ms != cs:
        mismatches.append((k, ms, cs))

if mismatches:
    print(f"\n{len(mismatches)} shape mismatches:")
    for k, ms, cs in mismatches[:20]:
        print(f"  {k}: model={ms}, ckpt={cs}")

if not missing and not extra and not mismatches:
    print("\nAll keys match! Loading checkpoint...")
    model.load_state_dict(ckpt_sd, strict=True)
    print("Checkpoint loaded successfully!")
elif not mismatches and not extra:
    print(f"\n{len(missing)} missing keys (may be buffers). Trying strict=False...")
    result = model.load_state_dict(ckpt_sd, strict=False)
    print(f"Missing keys: {result.missing_keys[:5]}")
    print(f"Unexpected keys: {result.unexpected_keys[:5]}")
