# CASSI InverseNet Validation Dataset

Reconstruction dataset for the CASSI (Coded Aperture Snapshot Spectral Imaging)
section of the InverseNet paper. Contains ground truth spectral cubes, masks,
measurements, and reconstructed spectral cubes for 3 scenarios x 4 methods x 10 scenes.

## Dataset Overview

| Item | Value |
|------|-------|
| Modality | CASSI (Snapshot Spectral Imaging) |
| Spatial resolution | 256 x 256 |
| Spectral bands | 28 per scene |
| Scenes | 10 (scene01 -- scene10, KAIST benchmark) |
| Methods | GAP-TV, HDNet, MST-S, MST-L |
| Scenarios | 3 (Ideal, Baseline, Oracle) |
| Total arrays | 170 (17 per scene x 10 scenes) |
| Data type | float32, range [0, 1] (spectral cubes) |

## Mismatch Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| mask_dx | 0.5 px | Horizontal sub-pixel shift |
| mask_dy | 0.3 px | Vertical sub-pixel shift |
| mask_theta | 0.1 deg | Rotation angle |
| noise_peak | 100000 | Poisson noise peak |
| noise_sigma | 0.01 | Gaussian noise std |

## CASSI Forward Model

The CASSI forward model includes spectral dispersion (step=2 pixels per band):

```
y[:, k*step : k*step + W] += mask * scene[:, :, k]   for k = 0..27
```

Measurement size: (256, 310) where 310 = 256 + (28-1) * 2.

## Scenarios

- **Scenario I (Ideal):** Ideal measurement (no mismatch, no noise) + ideal mask.
  Upper bound on reconstruction quality.

- **Scenario II (Baseline):** Corrupted measurement (warped mask + Poisson-Gaussian noise)
  + ideal mask (wrong operator). Shows degradation from operator mismatch.

- **Scenario III (Oracle):** Corrupted measurement (same as Scenario II)
  + true warped mask (oracle operator). Shows recovery when the true operator is known.

## Methods

| Method | Type | Reference |
|--------|------|-----------|
| GAP-TV | Optimization (GAP + Total Variation) | Yuan (2016) |
| HDNet | Dual-domain deep learning (mask-oblivious) | Hu et al. (2022) |
| MST-S | Mask-guided Spectral Transformer (small) | Cai et al. (2022) |
| MST-L | Mask-guided Spectral Transformer (large) | Cai et al. (2022) |

**Note:** HDNet is mask-oblivious (takes only the initial spectral estimate, not the mask
directly). Different masks still produce different initial estimates via `shift_back`,
so Scenario III results are identical to Scenario II for HDNet.

## File Structure

```
cassi_reconstructions/
    readme_cassi.md           # This file
    scene01.npz               # Scene 01 data
    scene02.npz               # Scene 02 data
    scene03.npz               # Scene 03 data
    scene04.npz               # Scene 04 data
    scene05.npz               # Scene 05 data
    scene06.npz               # Scene 06 data
    scene07.npz               # Scene 07 data
    scene08.npz               # Scene 08 data
    scene09.npz               # Scene 09 data
    scene10.npz               # Scene 10 data

../cassi_summary.json              # Aggregated metrics (per-method, overall)
../cassi_validation_results.json   # Per-scene detailed metrics (PSNR, SSIM, SAM)
```

## Array Keys per Scene

Each `.npz` file contains these arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `gt` | (256, 256, 28) | Ground truth spectral cube |
| `mask_ideal` | (256, 256) | Original binary coded-aperture mask |
| `mask_warped` | (256, 256) | Warped mask (shifted by dx, dy, theta) |
| `meas_ideal` | (256, 310) | Ideal measurement (no noise) |
| `meas_corrupt` | (256, 310) | Corrupted measurement with mismatch + noise |
| `scenario_i_gap_tv` | (256, 256, 28) | GAP-TV reconstruction, Scenario I |
| `scenario_i_hdnet` | (256, 256, 28) | HDNet reconstruction, Scenario I |
| `scenario_i_mst_s` | (256, 256, 28) | MST-S reconstruction, Scenario I |
| `scenario_i_mst_l` | (256, 256, 28) | MST-L reconstruction, Scenario I |
| `scenario_ii_gap_tv` | (256, 256, 28) | GAP-TV reconstruction, Scenario II |
| `scenario_ii_hdnet` | (256, 256, 28) | HDNet reconstruction, Scenario II |
| `scenario_ii_mst_s` | (256, 256, 28) | MST-S reconstruction, Scenario II |
| `scenario_ii_mst_l` | (256, 256, 28) | MST-L reconstruction, Scenario II |
| `scenario_iii_gap_tv` | (256, 256, 28) | GAP-TV reconstruction, Scenario III |
| `scenario_iii_hdnet` | (256, 256, 28) | HDNet reconstruction, Scenario III |
| `scenario_iii_mst_s` | (256, 256, 28) | MST-S reconstruction, Scenario III |
| `scenario_iii_mst_l` | (256, 256, 28) | MST-L reconstruction, Scenario III |

**17 arrays per scene x 10 scenes = 170 arrays total.**

## Usage Examples

### Load and inspect data

```python
import numpy as np

# Load one scene
d = np.load("cassi_reconstructions/scene01.npz")

# List all keys
print(sorted(d.keys()))

# Load ground truth and a reconstruction
gt = d["gt"]                            # (256, 256, 28) float32
recon_i = d["scenario_i_mst_l"]         # (256, 256, 28) float32

# Access a single spectral band (e.g., band 14)
band_gt = gt[:, :, 14]                  # (256, 256)
band_recon = recon_i[:, :, 14]          # (256, 256)
```

### Compute PSNR for a scene

```python
def psnr(x, y, max_val=1.0):
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(max_val ** 2 / mse)

gt = d["gt"]
recon = d["scenario_iii_mst_l"]
print(f"PSNR: {psnr(recon, gt):.2f} dB")
```

### Visual comparison across scenarios (one band)

```python
import matplotlib.pyplot as plt

d = np.load("cassi_reconstructions/scene01.npz")
band = 14  # pick a spectral band
method = "mst_l"

gt     = d["gt"][:, :, band]
sc_i   = d[f"scenario_i_{method}"][:, :, band]
sc_ii  = d[f"scenario_ii_{method}"][:, :, band]
sc_iii  = d[f"scenario_iii_{method}"][:, :, band]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, img, title in zip(axes,
    [gt, sc_i, sc_ii, sc_iii],
    ["Ground Truth", "Scenario I (Ideal)", "Scenario II (Baseline)", "Scenario III (Oracle)"]):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig("cassi_visual_comparison.png", dpi=150)
plt.show()
```

### Error map visualization

```python
gt = d["gt"][:, :, 14]
recon = d["scenario_ii_mst_l"][:, :, 14]
error = np.abs(gt - recon)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(gt, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Ground Truth")
axes[1].imshow(recon, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Scenario II Reconstruction")
axes[2].imshow(error, cmap="hot", vmin=0, vmax=0.3)
axes[2].set_title("Error Map")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("cassi_error_map.png", dpi=150)
```

### Compare all methods for one scenario

```python
d = np.load("cassi_reconstructions/scene05.npz")
band = 14
scenario = "scenario_i"
methods = ["gap_tv", "hdnet", "mst_s", "mst_l"]
labels = ["GAP-TV", "HDNet", "MST-S", "MST-L"]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(d["gt"][:, :, band], cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Ground Truth")
for ax, m, lab in zip(axes[1:], methods, labels):
    ax.imshow(d[f"{scenario}_{m}"][:, :, band], cmap="gray", vmin=0, vmax=1)
    ax.set_title(lab)
for ax in axes:
    ax.axis("off")
plt.suptitle(f"Scenario I -- Scene 05, Band {band}")
plt.tight_layout()
plt.savefig("cassi_method_comparison.png", dpi=150)
```

### Spectral profile comparison

```python
d = np.load("cassi_reconstructions/scene01.npz")
gt = d["gt"]
recon_i = d["scenario_i_mst_l"]
recon_ii = d["scenario_ii_mst_l"]
recon_iii = d["scenario_iii_mst_l"]

# Extract spectral profiles at pixel (128, 128)
pixel = (128, 128)
plt.figure(figsize=(8, 4))
plt.plot(gt[pixel[0], pixel[1], :], 'k-', label="Ground Truth", linewidth=2)
plt.plot(recon_i[pixel[0], pixel[1], :], 'b--', label="Scenario I (Ideal)")
plt.plot(recon_ii[pixel[0], pixel[1], :], 'r:', label="Scenario II (Baseline)")
plt.plot(recon_iii[pixel[0], pixel[1], :], 'g-.', label="Scenario III (Oracle)")
plt.xlabel("Spectral Band")
plt.ylabel("Intensity")
plt.title(f"Spectral Profile at pixel {pixel}")
plt.legend()
plt.tight_layout()
plt.savefig("cassi_spectral_profile.png", dpi=150)
```

### Load metrics from JSON

```python
import json

with open("../cassi_summary.json") as f:
    summary = json.load(f)

# Overall PSNR per method per scenario
for scenario in ["scenario_i", "scenario_ii", "scenario_iii"]:
    print(f"\n{scenario}:")
    for method, stats in summary[scenario].items():
        print(f"  {method}: {stats['psnr_mean']:.2f} +/- {stats['psnr_std']:.2f} dB  "
              f"SSIM={stats['ssim_mean']:.4f}  SAM={stats['sam_mean']:.2f} deg")
```

### Mask comparison (ideal vs warped)

```python
d = np.load("cassi_reconstructions/scene01.npz")
mask_ideal = d["mask_ideal"]
mask_warped = d["mask_warped"]
diff = np.abs(mask_ideal.astype(float) - mask_warped.astype(float))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(mask_ideal, cmap="gray")
axes[0].set_title("Ideal Mask")
axes[1].imshow(mask_warped, cmap="gray")
axes[1].set_title("Warped Mask")
axes[2].imshow(diff, cmap="hot")
axes[2].set_title(f"Difference ({diff.sum():.0f} total)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("cassi_mask_comparison.png", dpi=150)
```

## Validation Results Summary

| Method | Scenario I (Ideal) | Scenario II (Baseline) | Scenario III (Oracle) | Gap I-II | Recovery II-III |
|--------|-------------------|----------------------|---------------------|----------|--------------------|
| GAP-TV | 20.37 +/- 1.84 | 20.28 +/- 1.83 | 20.38 +/- 1.84 | +0.08 | +0.09 |
| HDNet | 34.66 +/- 2.62 | 24.05 +/- 1.85 | 24.05 +/- 1.85 | +10.61 | +0.00 |
| MST-S | 33.98 +/- 2.50 | 24.39 +/- 2.02 | 30.79 +/- 2.13 | +9.59 | +6.40 |
| MST-L | 34.81 +/- 2.11 | 24.23 +/- 1.97 | 32.22 +/- 2.04 | +10.58 | +7.99 |

All PSNR values in dB. Gap = degradation from mismatch. Recovery = improvement from oracle correction.

**Notes:**
- GAP-TV shows minimal degradation from mismatch (~0.08 dB) due to its iterative optimization nature.
- HDNet shows zero recovery (Scenario III = Scenario II) because it is mask-oblivious.
- MST-S and MST-L show strong recovery (+6.40 and +7.99 dB) as mask-aware transformers.

## Reproduction

To regenerate this dataset:

```bash
cd /home/spiritai/PWM/test2/Physics_World_Model
python papers/inversenet/scripts/validate_cassi_inversenet.py --device cuda --save-recon
```

Requires: PyTorch, scipy, numpy, and pretrained model checkpoints for HDNet,
MST-S, and MST-L (see validate_cassi_inversenet.py for paths).

## Source Data

The KAIST spectral scenes are loaded from `.mat` files via
`/home/spiritai/MST-main/datasets/TSA_simu_data/Truth/scene{01-10}.mat`.
Each scene is a 256x256x28 spectral cube (float32, range [0, 1]).
The mask is loaded from `/home/spiritai/MST-main/datasets/TSA_simu_data/mask.mat`.
