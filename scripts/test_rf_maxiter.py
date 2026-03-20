#!/usr/bin/env python3
"""Find max num_iter that works for ReconFormer on this machine."""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import h5py
import time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'reference', 'mri', 'reconformer'))
from Recurrent_Transformer import ReconFormer

ckpt = os.path.join(ROOT, 'reference', 'mri', 'reconformer_checkpoint.pth')

for num_iter in [2, 3, 4]:
    print(f"\n=== num_iter={num_iter} ===", flush=True)
    model = ReconFormer(
        in_channels=2, out_channels=2,
        num_ch=(96, 48, 24), num_iter=num_iter,
        down_scales=(2, 1, 1.5), img_size=320,
        num_heads=(6, 6, 6), depths=(2, 1, 1),
        window_sizes=(8, 8, 8), mlp_ratio=2.,
        resi_connection='1conv',
        use_checkpoint=[False]*6,
    )
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    model.eval()
    del state; gc.collect()

    x = torch.randn(1, 2, 320, 320)
    k0 = torch.randn(1, 2, 320, 320)
    mask = torch.ones(1, 1, 320, 320)

    t0 = time.time()
    sys.stdout.flush()
    try:
        with torch.no_grad():
            out = model(x, k0, mask)
        elapsed = time.time() - t0
        print(f"  OK: {out.shape}, {elapsed:.1f}s", flush=True)
    except Exception as e:
        print(f"  FAIL: {e}", flush=True)

    del model, x, k0, mask
    try:
        del out
    except:
        pass
    gc.collect()

print("\nDone.", flush=True)
