#!/usr/bin/env python3
"""Quick test of MiJUN on small synthetic input."""
import sys, os
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core', 'pwm_core', 'recon'))

import torch
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "reference", "cassi", "our_mamba_5stg_recovered.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    torch.cuda.empty_cache()
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test with 64x64 first (4x smaller)
from cassi_arch.MiJUN import MiJUN

# Build model with smaller spatial sizes for 64x64 input
model = MiJUN(stage=1, n_bands=28, step=2)  # 1 stage for speed

# Modify PP spatial sizes for 64x64
model.PP.spatial_sizes = (64, 32, 16)

state = torch.load(CKPT, map_location='cpu', weights_only=False)

# Load just the model weights (not mamba2 which depends on spatial size)
# Actually, mamba2 weights are fixed size — let's just test with wrong spatial size
# but correct param shapes to verify the forward pass works
try:
    model.load_state_dict(state['model'], strict=True)
    print("Checkpoint loaded.")
except Exception as e:
    print(f"Strict load failed: {e}")
    model.load_state_dict(state['model'], strict=False)
    print("Loaded with strict=False")

model = model.to(device).eval()

# Test forward pass with 64x64
# NOTE: mamba2 at encoder level 0 has d_model=256 (expects W=256)
# With 64x64, W=64, so mamba2 input shape won't match d_model=256
# This will fail — need to handle spatial-dependent mamba2 differently

# Let's try 256x256 but with only 1 stage
print("\nTesting 256x256 with 1 stage...")
H, W, nC = 256, 256, 28

mask = torch.ones(1, nC, H, W, device=device) * 0.5
y = torch.randn(1, H, W, device=device)
data = {'Y': y, 'mask': mask}

print("Running forward_test...")
t0 = time.time()
try:
    with torch.no_grad():
        out = model.forward_test(data)
    dt = time.time() - t0
    print(f"Output shape: {out.shape}")
    print(f"Time: {dt:.1f}s")
    print(f"Range: [{out.min().item():.4f}, {out.max().item():.4f}]")
except Exception as e:
    dt = time.time() - t0
    print(f"Failed after {dt:.1f}s: {e}")
    import traceback
    traceback.print_exc()
