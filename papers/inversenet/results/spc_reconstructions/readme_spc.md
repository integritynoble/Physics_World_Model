# SPC InverseNet Validation Dataset

Reconstruction dataset for the SPC (Single-Pixel Camera) section of the
InverseNet paper. Contains ground truth grayscale images and reconstructed
images for 3 scenarios x 3 methods x 11 test images.

## Dataset Overview

| Item | Value |
|------|-------|
| Modality | SPC (Single-Pixel Camera / Compressive Sensing) |
| Images | 11 (Set11 benchmark) |
| Compression ratio | 25% (272 measurements / 1089 pixels per 33x33 block) |
| Methods | FISTA-TV, ISTA-Net, HATNet |
| Scenarios | 3 (Ideal, Baseline, Corrected) |
| Total arrays | 110 (10 per image x 11 images) |
| Data type | float32, pixel range [0, 255] |

## Mismatch Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| gain_alpha | 0.0015 | Exponential gain drift rate (ISTA/FISTA) |
| gain_alpha_h | 0.0015 | 2D gain drift rate, rows (HATNet) |
| gain_alpha_w | 0.0015 | 2D gain drift rate, cols (HATNet) |
| sigma_y | 0.03 | Gaussian noise std (ISTA/FISTA) |
| sigma_y_hat | 0.04 | Sensor noise std (HATNet) |

## Mismatch Model

**Gain drift:** Per-row exponential decay applied to measurements:
```
g_i = exp(-alpha * i)    for i = 0, 1, ..., m-1
```

For ISTA-Net/FISTA-TV (1D): gain vector g applied to 33x33 block measurements.
For HATNet (2D): separable gain matrix G = outer(g_h, g_w) applied to 128x128 measurements.

## Scenarios

- **Scenario I (Ideal):** Clean measurement (no gain drift, no noise*) + ideal operator.
  Upper bound on reconstruction quality.

- **Scenario II (Baseline):** Gain-drifted + noisy measurement + assumed ideal operator.
  Shows degradation from operator mismatch.

- **Scenario III (Corrected):** Corrected measurement (y / gain) + assumed ideal operator.
  Shows recovery when the gain drift is known and corrected.

*For HATNet, Scenario I includes sensor noise to match ~30.78 dB target.

## Methods

| Method | Type | Block size | Reference |
|--------|------|------------|-----------|
| FISTA-TV | Classical (FISTA + TV proximal) | 33x33 | Beck & Teboulle (2009) |
| ISTA-Net | Deep unfolding (9 layers, learned) | 33x33 | Zhang & Ghanem (2018) |
| HATNet | Hybrid Attention Transformer | 256x256 | Full-image, Kronecker sampling |

**Note:** ISTA-Net and FISTA-TV share the same learned sampling matrix Phi (272x1089)
and operate on 33x33 pixel blocks. HATNet uses Kronecker-structured measurement
matrices H (128x256) and W (128x256) and operates on full 256x256 images.

## File Structure

```
spc_reconstructions/
    readme_spc.md             # This file
    Monarch.npz               # 768x512 (processed as 4 quadrants for HATNet)
    Parrots.npz               # 768x512
    barbara.npz               # 720x576
    boats.npz                 # 576x720
    cameraman.npz             # 256x256
    fingerprint.npz           # 512x512
    flinstones.npz            # 720x576
    foreman.npz               # 352x288
    house.npz                 # 256x256
    lena256.npz               # 256x256
    peppers256.npz            # 256x256

../spc_summary.json                # Aggregated metrics (per-method, overall)
../spc_validation_results.json     # Per-image detailed metrics (PSNR, SSIM)
```

## Array Keys per Image

Each `.npz` file contains these arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `gt` | (H, W) | Ground truth grayscale image (0-255 range) |
| `scenario_i_fista_tv` | (H, W) | FISTA-TV reconstruction, Scenario I |
| `scenario_i_ista_net` | (H, W) | ISTA-Net reconstruction, Scenario I |
| `scenario_i_hatnet` | (H, W) | HATNet reconstruction, Scenario I |
| `scenario_ii_fista_tv` | (H, W) | FISTA-TV reconstruction, Scenario II |
| `scenario_ii_ista_net` | (H, W) | ISTA-Net reconstruction, Scenario II |
| `scenario_ii_hatnet` | (H, W) | HATNet reconstruction, Scenario II |
| `scenario_iii_fista_tv` | (H, W) | FISTA-TV reconstruction, Scenario III |
| `scenario_iii_ista_net` | (H, W) | ISTA-Net reconstruction, Scenario III |
| `scenario_iii_hatnet` | (H, W) | HATNet reconstruction, Scenario III |

**10 arrays per image x 11 images = 110 arrays total.**

All pixel values are float32 in range [0, 255].

## Usage Examples

### Load and inspect data

```python
import numpy as np

# Load one image
d = np.load("spc_reconstructions/lena256.npz")

# List all keys
print(sorted(d.keys()))

# Load ground truth and a reconstruction
gt = d["gt"]                                # (256, 256) float32
recon_i = d["scenario_i_ista_net"]          # (256, 256) float32

print(f"GT shape: {gt.shape}, range: [{gt.min():.0f}, {gt.max():.0f}]")
```

### Compute PSNR

```python
def psnr_255(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))

gt = d["gt"]
recon = d["scenario_iii_ista_net"]
print(f"PSNR: {psnr_255(recon, gt):.2f} dB")
```

### Visual comparison across scenarios

```python
import matplotlib.pyplot as plt

d = np.load("spc_reconstructions/cameraman.npz")
method = "ista_net"

gt     = d["gt"]
sc_i   = d[f"scenario_i_{method}"]
sc_ii  = d[f"scenario_ii_{method}"]
sc_iii = d[f"scenario_iii_{method}"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, img, title in zip(axes,
    [gt, sc_i, sc_ii, sc_iii],
    ["Ground Truth", "Scenario I (Ideal)", "Scenario II (Baseline)", "Scenario III (Corrected)"]):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig("spc_visual_comparison.png", dpi=150)
plt.show()
```

### Compare all methods for one scenario

```python
d = np.load("spc_reconstructions/lena256.npz")
scenario = "scenario_i"
methods = ["fista_tv", "ista_net", "hatnet"]
labels = ["FISTA-TV", "ISTA-Net", "HATNet"]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(d["gt"], cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Ground Truth")
for ax, m, lab in zip(axes[1:], methods, labels):
    ax.imshow(d[f"{scenario}_{m}"], cmap="gray", vmin=0, vmax=255)
    ax.set_title(lab)
for ax in axes:
    ax.axis("off")
plt.suptitle("Scenario I -- lena256")
plt.tight_layout()
plt.savefig("spc_method_comparison.png", dpi=150)
```

### Error map visualization

```python
d = np.load("spc_reconstructions/cameraman.npz")
gt = d["gt"]
recon = d["scenario_ii_ista_net"]
error = np.abs(gt - recon)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(gt, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Ground Truth")
axes[1].imshow(recon, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Scenario II Reconstruction")
axes[2].imshow(error, cmap="hot", vmin=0, vmax=50)
axes[2].set_title("Error Map")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("spc_error_map.png", dpi=150)
```

### Load metrics from JSON

```python
import json

with open("../spc_summary.json") as f:
    summary = json.load(f)

methods = ["fista_tv", "ista_net", "hatnet"]
for method in methods:
    print(f"\n{method}:")
    for scenario in ["scenario_i", "scenario_ii", "scenario_iii"]:
        key = f"{method}_{scenario}"
        s = summary["methods"][key]
        print(f"  {scenario}: {s['psnr_mean']:.2f} +/- {s['psnr_std']:.2f} dB  "
              f"SSIM={s['ssim_mean']:.4f}")
```

## Validation Results Summary

| Method | Scenario I (Ideal) | Scenario II (Baseline) | Scenario III (Corrected) | Gap I-II | Recovery II-III |
|--------|-------------------|----------------------|------------------------|----------|-----------------|
| FISTA-TV | 28.06 +/- 3.38 | 18.51 +/- 0.69 | 26.21 +/- 2.28 | +9.55 | +7.71 |
| ISTA-Net | 31.85 +/- 3.11 | 19.02 +/- 0.61 | 27.45 +/- 1.32 | +12.83 | +8.43 |
| HATNet | 30.98 +/- 0.95 | 19.40 +/- 0.59 | 29.78 +/- 0.81 | +11.58 | +10.38 |

All PSNR values in dB (255-scale). Gap = degradation from mismatch. Recovery = improvement from correction.

## Reproduction

To regenerate this dataset:

```bash
cd /home/spiritai/PWM/test2/Physics_World_Model
python papers/inversenet/scripts/validate_spc_inversenet.py --save-recon
```

Requires: PyTorch, scipy, numpy, opencv-python, scikit-image, and pretrained model
checkpoints for ISTA-Net and HATNet (see validate_spc_inversenet.py for paths).

## Source Data

Test images (Set11 benchmark) loaded from:
`/home/spiritai/ISTA-Net-PyTorch-master/data/Set11/*.tif`

11 grayscale images of varying sizes (256x256 to 768x576).
ISTA-Net sampling matrix: `/home/spiritai/ISTA-Net-PyTorch-master/sampling_matrix/phi_0_25_1089.mat`
