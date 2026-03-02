# 05 — Hands-On Tutorial

This tutorial walks through loading MRI benchmark data, running
reconstruction algorithms, computing metrics, and visualising results.
All code is runnable from the repository root.

## Setup

```python
import sys
from pathlib import Path

# Add project paths
ROOT = Path("/home/spiritai/abraham/pwm/production/Physics_World_Model")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import numpy as np
import h5py
import json
```

---

## 1. Loading an HDF5 Sample

```python
h5_path = ROOT / "datasets/benchmark/mri/public/mri_challenge_public.h5"

with h5py.File(h5_path, "r") as hf:
    # List all samples
    sample_keys = sorted([k for k in hf.keys() if k.startswith("sample_")])
    print(f"Found {len(sample_keys)} samples: {sample_keys[:3]}...")

    # Load first sample
    grp = hf["sample_00"]

    x_true    = grp["x_true"][:]       # (320, 320) float32, range [0, 1]
    y_kspace  = grp["y_kspace"][:]     # (15, 320, 320) complex64
    mask      = grp["mask"][:]         # (320,) uint8, values {0, 1}
    coil_maps = grp["coil_maps"][:]    # (15, 320, 320) complex64
    b0_map    = grp["B0_map"][:]       # (320, 320) float32, range [-1, 1]
    warp      = grp["warp_field"][:]   # (2, 320, 320) float32

    # Parse metadata
    metadata  = json.loads(grp.attrs["metadata"])
    true_spec = json.loads(grp.attrs["true_spec"])

print(f"Image shape:     {x_true.shape}, dtype={x_true.dtype}")
print(f"K-space shape:   {y_kspace.shape}, dtype={y_kspace.dtype}")
print(f"Mask shape:      {mask.shape}, sampled lines: {mask.sum()}/{len(mask)}")
print(f"Coil maps shape: {coil_maps.shape}")
print(f"Scene:           {metadata.get('scene', 'unknown')}")
print(f"Mismatch:        B0={true_spec['B0_inhomog_hz']:.1f} Hz, "
      f"noise={true_spec['noise_sigma']:.3f}")
```

---

## 2. Visualising the Data

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Ground truth
axes[0, 0].imshow(x_true, cmap="gray")
axes[0, 0].set_title("Ground Truth (x_true)")

# Undersampling mask
axes[0, 1].imshow(mask.reshape(-1, 1) * np.ones((1, 320)),
                  cmap="gray", aspect="auto")
axes[0, 1].set_title(f"Mask (R={320 // mask.sum():.0f}x)")

# K-space magnitude (log scale, first coil)
kspace_mag = np.log1p(np.abs(y_kspace[0]))
axes[0, 2].imshow(kspace_mag, cmap="viridis")
axes[0, 2].set_title("K-space (coil 0, log)")

# Coil sensitivity maps (magnitude, first 3 coils)
for i in range(3):
    axes[1, i].imshow(np.abs(coil_maps[i]), cmap="hot")
    axes[1, i].set_title(f"Coil {i} sensitivity")

for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.savefig("mri_data_overview.png", dpi=150)
plt.show()
```

---

## 3. Running Zero-Filled RSS (Baseline)

```python
from pwm_core.recon.mri_solvers import zero_filled_reconstruction

# Multi-coil input -> RSS output
x_zf = zero_filled_reconstruction(y_kspace, device=None)
# Returns: (320, 320) float32

# Normalise to [0, 1]
x_zf = x_zf / (x_zf.max() + 1e-10)

print(f"Zero-filled shape: {x_zf.shape}, range: [{x_zf.min():.3f}, {x_zf.max():.3f}]")
```

---

## 4. Running SENSE

```python
from pwm_core.recon.mri_solvers import sense_reconstruction

# SENSE needs a 2D mask: expand 1D mask (320,) -> (320, 320)
H, W = 320, 320
mask_2d = mask.astype(np.float32).reshape(-1, 1) * np.ones((1, W), dtype=np.float32)

x_sense = sense_reconstruction(
    kspace=y_kspace,             # (15, 320, 320) complex64
    sensitivity_maps=coil_maps,  # (15, 320, 320) complex64
    mask=mask_2d,                # (320, 320) float32
    regularization=0.001,
    iterations=30,
    device=None
)
# Returns: (320, 320) complex64

# Take magnitude and normalise
x_sense = np.abs(x_sense).astype(np.float32)
x_sense = x_sense / (x_sense.max() + 1e-10)

print(f"SENSE shape: {x_sense.shape}")
```

---

## 5. Running CS-MRI

```python
from pwm_core.recon.mri_solvers import cs_mri_wavelet

x_cs = cs_mri_wavelet(
    kspace=y_kspace,              # (15, 320, 320) — multi-coil
    mask=mask_2d,                 # (320, 320)
    lam=0.01,                     # sparsity weight
    iterations=50,
    sensitivity_maps=coil_maps,   # provide to avoid auto-estimation
    device=None
)
# Returns: (320, 320) complex64

x_cs = np.abs(x_cs).astype(np.float32)
x_cs = x_cs / (x_cs.max() + 1e-10)
```

---

## 6. Running PnP-HQS

PnP requires explicit forward/adjoint operators. We build them from the
coil maps and mask.

```python
from pwm_core.recon.pnp import pnp_hqs, get_denoiser
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Build forward/adjoint operators
class MultiCoilMRIOp:
    def __init__(self, coil_maps, mask_1d):
        self.coil_maps = coil_maps  # (C, H, W)
        H, W = coil_maps.shape[1], coil_maps.shape[2]
        self.mask_2d = mask_1d.astype(np.float32).reshape(-1, 1) \
                       * np.ones((1, W), dtype=np.float32)

    def forward(self, x):
        """x: (H,W) -> y: (C,H,W) complex"""
        x_c = x.astype(np.complex64)
        y = np.zeros_like(self.coil_maps)
        for c in range(self.coil_maps.shape[0]):
            img_c = self.coil_maps[c] * x_c
            y[c] = self.mask_2d * fftshift(fft2(ifftshift(img_c)))
        return y

    def adjoint(self, y):
        """y: (C,H,W) -> x: (H,W) float32"""
        x = np.zeros(y.shape[1:], dtype=np.complex64)
        for c in range(y.shape[0]):
            k_masked = self.mask_2d * y[c]
            img_c = fftshift(ifft2(ifftshift(k_masked)))
            x += np.conj(self.coil_maps[c]) * img_c
        return np.abs(x).astype(np.float32)

op = MultiCoilMRIOp(coil_maps, mask)
denoiser = get_denoiser("auto", device="cpu")

x_pnp = pnp_hqs(
    y=y_kspace,
    forward=op.forward,
    adjoint=op.adjoint,
    x_shape=(320, 320),
    denoiser=denoiser,
    iters=30,
    rho=1.0,
    sigma=0.1,
    sigma_decay=0.9
)
# Returns: (320, 320) float32

x_pnp = x_pnp / (x_pnp.max() + 1e-10)
```

---

## 7. Running VarNet (Single-Coil)

VarNet requires single-coil input. We combine multi-coil k-space via RSS.

```python
from pwm_core.recon.varnet import varnet_recon

# Coil-combine to single-coil k-space
imgs = np.fft.ifft2(np.fft.ifftshift(y_kspace, axes=(-2,-1)), axes=(-2,-1))
rss_img = np.sqrt(np.sum(np.abs(imgs)**2, axis=0))
kspace_combined = np.fft.fftshift(np.fft.fft2(rss_img))

x_varnet = varnet_recon(
    kspace=kspace_combined,   # (320, 320) complex
    mask=mask.astype(np.float32),  # (320,) -> handled internally
    n_cascades=12,
    device=None               # auto: cuda if available
)
# Returns: (320, 320) float32
# Note: without pretrained weights, results ≈ zero-filled

x_varnet = x_varnet / (x_varnet.max() + 1e-10)
```

---

## 8. Running MoDL (Single-Coil)

```python
from pwm_core.recon.modl import modl_recon

x_modl = modl_recon(
    kspace=kspace_combined,   # (320, 320) complex
    mask=mask.astype(np.float32),  # (320,) -> tiled to (320, 320) internally
    n_iter=5,
    device=None
)
# Returns: (320, 320) float32
# Note: without pretrained weights, results ≈ zero-filled

x_modl = x_modl / (x_modl.max() + 1e-10)
```

---

## 9. Computing Metrics

```python
from benchmarks.framework.metrics import compute_psnr, compute_ssim

results = {}
recons = {
    "Zero-Filled RSS": x_zf,
    "SENSE": x_sense,
    "CS-MRI": x_cs,
    "PnP-HQS": x_pnp,
    "VarNet": x_varnet,
    "MoDL": x_modl,
}

print(f"\n{'Algorithm':<20} {'PSNR (dB)':>10} {'SSIM':>8}")
print("-" * 40)

for name, x_hat in recons.items():
    psnr = compute_psnr(x_true, x_hat, max_val=1.0)
    ssim = compute_ssim(x_true, x_hat, data_range=1.0)
    results[name] = {"psnr": psnr, "ssim": ssim}
    print(f"{name:<20} {psnr:>10.2f} {ssim:>8.4f}")
```

Expected output (approximate, public tier):
```
Algorithm                PSNR (dB)     SSIM
----------------------------------------
Zero-Filled RSS           25.30   0.7200
SENSE                     31.20   0.8800
CS-MRI                    30.50   0.8600
PnP-HQS                  30.80   0.8700
VarNet                    24.10   0.6900
MoDL                      23.80   0.6800
```

---

## 10. Visualising Reconstructions

```python
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Top row: reconstructions
images = [x_true, x_zf, x_sense, x_cs]
titles = ["Ground Truth", "Zero-Filled RSS", "SENSE", "CS-MRI"]
for ax, img, title in zip(axes[0], images, titles):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

# Bottom row: more reconstructions + error maps
images2 = [x_pnp, x_varnet, x_modl, np.abs(x_true - x_sense)]
titles2 = ["PnP-HQS", "VarNet (random)", "MoDL (random)", "SENSE Error (×5)"]
for ax, img, title in zip(axes[1], images2, titles2):
    if "Error" in title:
        ax.imshow(img * 5, cmap="hot", vmin=0, vmax=0.5)
    else:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig("mri_reconstructions.png", dpi=150)
plt.show()
```

---

## 11. Comparing Across Tiers

```python
# Run the same algorithm on all tiers to see mismatch impact
tiers = {
    "public": ROOT / "datasets/benchmark/mri/public/mri_challenge_public.h5",
    "dev":    ROOT / "datasets/benchmark/mri/dev/mri_challenge_dev.h5",
    "hidden": ROOT / "datasets/benchmark/mri/hidden/mri_challenge_hidden.h5",
}

for tier_name, h5_path in tiers.items():
    with h5py.File(h5_path, "r") as hf:
        grp = hf["sample_00"]
        xt = grp["x_true"][:]
        yk = grp["y_kspace"][:]

    x_hat = zero_filled_reconstruction(yk, device=None)
    x_hat = x_hat / (x_hat.max() + 1e-10)

    psnr = compute_psnr(xt, x_hat, max_val=1.0)
    ssim = compute_ssim(xt, x_hat, data_range=1.0)
    print(f"{tier_name:>8}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
```

You should see PSNR decrease from public → dev → hidden, confirming
the mismatch severity progression.

---

## 12. Running the Full Benchmark

```bash
# Quick test (2 samples, public tier, zero-filled only)
python papers/pwm_flagship/scripts/run_mri_multiphantom.py \
    --tier public --solver zerofilled --max-samples 2

# Full benchmark (all tiers, all solvers)
python papers/pwm_flagship/scripts/run_mri_multiphantom.py --tier all --solver all
```

Results are saved to:
`papers/pwm_flagship/results/real_data_4scenario/mri_multiphantom_results.json`

---

*Previous: [04 — PWM MRI Benchmark](04_pwm_mri_benchmark.md)*
*Back to: [README](README.md)*
