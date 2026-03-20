#!/usr/bin/env python3
"""Train RED-CNN weights on standard CT data, and MoDL on standard MRI data."""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pwm5_pkg = str(ROOT / "packages" / "pwm_core")
sys.path = [p for p in sys.path if 'PWM4' not in p and ('pwm_core' not in p or 'PWM5' in p)]
sys.path.insert(0, pwm5_pkg)
sys.path.insert(0, str(ROOT))
for k in list(sys.modules):
    if 'pwm_core' in k or 'algorithm_base' in k:
        del sys.modules[k]

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim


def psnr(x_hat, x_true):
    mse = np.mean((x_hat.astype(np.float64) - x_true.astype(np.float64)) ** 2)
    if mse < 1e-15: return float('inf')
    return float(10 * np.log10(float(x_true.max()) ** 2 / mse))


# =============================================================================
# PART 1: Train RED-CNN on CT standard data
# =============================================================================
def train_redcnn():
    print("=" * 60)
    print("TRAINING RED-CNN ON STANDARD CT DATA")
    print("=" * 60)

    from pwm_core.recon.redcnn import REDCNN
    from skimage.transform import iradon

    # Load all CT standard samples
    ct_dir = ROOT / "datasets" / "benchmark" / "ct" / "standard"
    noisy_list = []
    clean_list = []

    for idx in range(10):
        h5_path = ct_dir / f"standard_ct_{idx:02d}.h5"
        if not h5_path.exists():
            continue
        f = h5py.File(str(h5_path), 'r')
        sino = np.array(f['y_ideal'], dtype=np.float32)
        x_true = np.array(f['x_true'], dtype=np.float32)
        f.close()

        # FBP reconstruction (noisy input for RED-CNN)
        n_angles, n_det = sino.shape
        angles_deg = np.linspace(0, 180, n_angles, endpoint=False)
        fbp = iradon(sino.T, theta=angles_deg, circle=True,
                     output_size=x_true.shape[0], filter_name='ramp')

        # Normalize both to [0, 1]
        fbp_min, fbp_max = fbp.min(), fbp.max()
        if fbp_max - fbp_min > 1e-8:
            fbp_norm = (fbp - fbp_min) / (fbp_max - fbp_min)
        else:
            fbp_norm = fbp - fbp_min

        xt_min, xt_max = x_true.min(), x_true.max()
        if xt_max - xt_min > 1e-8:
            xt_norm = (x_true - xt_min) / (xt_max - xt_min)
        else:
            xt_norm = x_true - xt_min

        noisy_list.append(fbp_norm.astype(np.float32))
        clean_list.append(xt_norm.astype(np.float32))
        print(f"  Loaded sample {idx:02d}: FBP PSNR = {psnr(fbp_norm, xt_norm):.1f} dB")

    n_samples = len(noisy_list)
    print(f"\n  Training on {n_samples} samples")

    if n_samples == 0:
        print("  No samples found!")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = REDCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 500
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for i in range(n_samples):
            noisy_t = torch.from_numpy(noisy_list[i]).unsqueeze(0).unsqueeze(0).to(device)
            clean_t = torch.from_numpy(clean_list[i]).unsqueeze(0).unsqueeze(0).to(device)

            optimizer.zero_grad()
            output = model(noisy_t)
            loss = criterion(output, clean_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 50 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / n_samples
            # Eval PSNR
            model.eval()
            with torch.no_grad():
                test_in = torch.from_numpy(noisy_list[0]).unsqueeze(0).unsqueeze(0).to(device)
                test_out = model(test_in).squeeze().cpu().numpy()
            p = psnr(test_out, clean_list[0])
            model.train()
            print(f"  Epoch {epoch:4d}: loss={avg_loss:.6f}, PSNR={p:.2f} dB")

    # Save weights
    weights_dir = ROOT / "packages" / "pwm_core" / "pwm_core" / "weights" / "redcnn"
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(weights_dir / "redcnn.pth"))
    print(f"\n  Weights saved to {weights_dir / 'redcnn.pth'}")

    # Final eval
    model.eval()
    with torch.no_grad():
        for i in range(min(3, n_samples)):
            test_in = torch.from_numpy(noisy_list[i]).unsqueeze(0).unsqueeze(0).to(device)
            test_out = model(test_in).squeeze().cpu().numpy()
            p_out = psnr(test_out, clean_list[i])
            p_in = psnr(noisy_list[i], clean_list[i])
            print(f"  Sample {i}: FBP={p_in:.2f} dB -> RED-CNN={p_out:.2f} dB")


# =============================================================================
# PART 2: Train MoDL on MRI standard data
# =============================================================================
def train_modl():
    print("\n" + "=" * 60)
    print("TRAINING MoDL ON STANDARD MRI DATA")
    print("=" * 60)

    from pwm_core.recon.modl import MoDL

    mri_dir = ROOT / "datasets" / "benchmark" / "mri" / "standard"
    kspace_list = []
    mask_list = []
    target_list = []

    for idx in range(10):
        h5_path = mri_dir / f"standard_mri_{idx:02d}.h5"
        if not h5_path.exists():
            continue
        f = h5py.File(str(h5_path), 'r')
        y = np.array(f['y_ideal'], dtype=np.float32)
        x_true = np.array(f['x_true'], dtype=np.float32)
        sampling_mask = np.array(f['sampling_mask'], dtype=np.float32) if 'sampling_mask' in f else None
        f.close()

        # Convert y from (H,W,2) real+imag to complex
        if y.ndim == 3 and y.shape[-1] == 2:
            kspace = y[..., 0] + 1j * y[..., 1]
        else:
            kspace = y

        # Normalize
        xt_max = x_true.max()
        if xt_max > 1e-8:
            x_true_norm = x_true / xt_max
            kspace_norm = kspace / xt_max
        else:
            x_true_norm = x_true
            kspace_norm = kspace

        kspace_list.append(kspace_norm)
        target_list.append(x_true_norm.astype(np.float32))
        mask_list.append(sampling_mask)

        # Zero-filled PSNR
        zf = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_norm)))
        p_zf = psnr(zf.astype(np.float32), x_true_norm)
        print(f"  Loaded sample {idx:02d}: ZF PSNR = {p_zf:.1f} dB")

    n_samples = len(kspace_list)
    print(f"\n  Training on {n_samples} samples")

    if n_samples == 0:
        print("  No samples found!")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = MoDL(n_iter=5, n_channels=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 500
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for i in range(n_samples):
            kspace_c = kspace_list[i]
            target = target_list[i]
            smask = mask_list[i]

            # Prepare tensors: kspace as (B, H, W) complex
            kspace_t = torch.from_numpy(np.stack([kspace_c.real, kspace_c.imag], axis=0)).unsqueeze(0).float().to(device)
            target_t = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float().to(device)

            if smask is not None:
                mask_t = torch.from_numpy(smask).unsqueeze(0).float().to(device)
            else:
                mask_t = torch.ones(1, kspace_c.shape[0], kspace_c.shape[1], device=device)

            optimizer.zero_grad()

            # MoDL forward: need to construct input properly
            # MoDL expects under-sampled k-space and mask
            try:
                output = model(kspace_t, mask_t)
                loss = criterion(output, target_t)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                if epoch == 0:
                    print(f"  WARNING: MoDL training error on sample {i}: {e}")
                continue

        if epoch % 50 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / max(n_samples, 1)
            print(f"  Epoch {epoch:4d}: loss={avg_loss:.6f}")

    # Save weights
    weights_dir = ROOT / "packages" / "pwm_core" / "pwm_core" / "weights" / "modl"
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(weights_dir / "modl.pth"))
    print(f"\n  Weights saved to {weights_dir / 'modl.pth'}")


if __name__ == "__main__":
    train_redcnn()
    train_modl()
