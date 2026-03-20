#!/usr/bin/env python3
"""Pinpoint where ReconFormer segfaults."""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'reference', 'mri', 'reconformer'))

print("1. Import...", flush=True)
from Recurrent_Transformer import ReconFormer, TransBlock_UC, TransBlock_OC, DataConsistencyInKspace
print("   OK", flush=True)

print("2. Create model (num_iter=1)...", flush=True)
model = ReconFormer(
    in_channels=2, out_channels=2,
    num_ch=(96, 48, 24), num_iter=1,
    down_scales=(2, 1, 1.5), img_size=320,
    num_heads=(6, 6, 6), depths=(2, 1, 1),
    window_sizes=(8, 8, 8), mlp_ratio=2.,
    resi_connection='1conv',
    use_checkpoint=[False]*6,
)
print("   OK", flush=True)

print("3. Load checkpoint...", flush=True)
ckpt = os.path.join(ROOT, 'reference', 'mri', 'reconformer_checkpoint.pth')
state = torch.load(ckpt, map_location='cpu', weights_only=False)
model.load_state_dict(state)
model.eval()
del state; gc.collect()
print("   OK", flush=True)

# Test with small dummy input first (64x64)
for size in [64, 128, 160, 320]:
    print(f"4. Forward pass (CPU, {size}x{size})...", flush=True)
    sys.stdout.flush()
    try:
        x = torch.randn(1, 2, size, size)
        k0 = torch.randn(1, 2, size, size)
        mask = torch.ones(1, 1, size, size)
        t0 = time.time()
        with torch.no_grad():
            out = model(x, k0, mask)
        elapsed = time.time() - t0
        print(f"   OK: {out.shape}, {elapsed:.1f}s", flush=True)
        del x, k0, mask, out
        gc.collect()
    except Exception as e:
        print(f"   FAIL: {e}", flush=True)
        break

print("Done.", flush=True)
