#!/usr/bin/env python3
"""Test pretrained ReconFormer loading and inference."""
import sys, os, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'pwm_core'))

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "reference", "mri")

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# Step 1: Try loading model architecture
print("\n--- Step 1: Load model architecture ---")
try:
    reconformer_dir = os.path.join(CKPT_DIR, "reconformer")
    sys.path.insert(0, reconformer_dir)
    from Recurrent_Transformer import ReconFormer
    print("  [OK] ReconFormer class imported")
except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Instantiate model
print("\n--- Step 2: Instantiate model ---")
try:
    model = ReconFormer(
        in_channels=2, out_channels=2,
        num_ch=(96, 48, 24), num_iter=5,
        down_scales=(2, 1, 1.5), img_size=320,
        num_heads=(6, 6, 6), depths=(2, 1, 1),
        window_sizes=(8, 8, 8), mlp_ratio=2.,
        resi_connection='1conv',
        use_checkpoint=[False]*6,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] Model created: {n_params:,} parameters ({n_params/1e6:.1f}M)")
except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 3: Load checkpoint
print("\n--- Step 3: Load checkpoint ---")
try:
    ckpt_path = os.path.join(CKPT_DIR, "reconformer_checkpoint.pth")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"  [OK] Checkpoint loaded from {ckpt_path}")
except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test inference with dummy data
print("\n--- Step 4: Dummy inference (CPU) ---")
try:
    x = torch.randn(1, 2, 320, 320)
    k0 = torch.randn(1, 2, 320, 320)
    mask = torch.ones(1, 1, 320, 320)
    with torch.no_grad():
        out = model(x, k0, mask)
    print(f"  [OK] Output shape: {out.shape}")
except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test with real data
print("\n--- Step 5: Real data inference ---")
try:
    import h5py
    DATA_DIR = os.path.join(ROOT, "datasets", "benchmark", "mri", "standard")
    h5_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.h5')])
    fname = h5_files[0]
    with h5py.File(os.path.join(DATA_DIR, fname), 'r') as hf:
        x_true = np.array(hf['x_true'], dtype=np.float32)
        y_raw = np.array(hf['y_ideal'], dtype=np.float32)
        mask_data = np.array(hf['sampling_mask'], dtype=np.float32)

    # Complex kspace
    from scipy.fft import ifft2, ifftshift
    kspace = (y_raw[..., 0] + 1j * y_raw[..., 1]).astype(np.complex64)

    # Zero-filled image
    zf = ifft2(ifftshift(kspace))
    zf_2ch = np.stack([zf.real.astype(np.float32), zf.imag.astype(np.float32)], axis=-1)

    # Normalize
    mag = np.sqrt(zf_2ch[..., 0]**2 + zf_2ch[..., 1]**2)
    std_val = float(mag.mean()) + 1e-11
    print(f"  std_val = {std_val:.6f}")

    img_t = torch.from_numpy(zf_2ch).permute(2, 0, 1).unsqueeze(0).float() / std_val
    k0_2ch = np.stack([kspace.real.astype(np.float32), kspace.imag.astype(np.float32)], axis=-1)
    k0_t = torch.from_numpy(k0_2ch).permute(2, 0, 1).unsqueeze(0).float() / std_val
    mask_t = torch.from_numpy(mask_data).unsqueeze(0).unsqueeze(0).float()

    print(f"  img_t: {img_t.shape}, k0_t: {k0_t.shape}, mask_t: {mask_t.shape}")

    with torch.no_grad():
        out = model(img_t, k0_t, mask_t)

    out_np = out.squeeze().numpy()  # (2, H, W)
    result = np.sqrt(out_np[0]**2 + out_np[1]**2).astype(np.float32) * std_val

    mse = np.mean((x_true - result)**2)
    psnr_val = 10 * np.log10(1.0 / mse) if mse > 1e-10 else 100.0
    print(f"  [OK] PSNR = {psnr_val:.2f} dB")
    print(f"  result range: [{result.min():.4f}, {result.max():.4f}]")
    print(f"  x_true range: [{x_true.min():.4f}, {x_true.max():.4f}]")
except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()

print("\nDone.")
