#!/usr/bin/env python3
"""List all model keys grouped by top-level module."""
import torch, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "reference", "cassi", "our_mamba_5stg_recovered.pth")
state = torch.load(CKPT, map_location='cpu', weights_only=False)
model_state = state['model']

# Group by module
groups = defaultdict(list)
for k in sorted(model_state.keys()):
    # Group by first 3 dot-separated parts
    parts = k.split('.')
    if parts[0] in ('fusion',):
        group = 'fusion'
    elif len(parts) >= 3:
        group = '.'.join(parts[:3])
    else:
        group = '.'.join(parts[:2])
    v = model_state[k]
    shape = list(v.shape) if hasattr(v, 'shape') else str(v)
    groups[group].append((k, shape))

for g in sorted(groups.keys()):
    print(f"\n=== {g} ===")
    for k, s in groups[g]:
        print(f"  {k}: {s}")

# Also check Mamba dims
print("\n\n=== Mamba dimensions ===")
for k in sorted(model_state.keys()):
    if 'mamba' in k and 'in_proj' in k:
        v = model_state[k]
        d_inner, d_model = v.shape
        prefix = k.rsplit('.in_proj', 1)[0]
        x_proj = model_state.get(f"{prefix}.x_proj.weight")
        dt_proj = model_state.get(f"{prefix}.dt_proj.weight")
        A_log = model_state.get(f"{prefix}.A_log")
        if x_proj is not None:
            dt_rank = x_proj.shape[0] - 2 * A_log.shape[1]
        else:
            dt_rank = "?"
        print(f"  {prefix}: d_model={d_model}, d_inner={d_inner//2}, d_state={A_log.shape[1] if A_log is not None else '?'}, dt_rank={dt_rank}")
