# CACTI InverseNet Validation Dataset

Reconstruction dataset for the CACTI (Coded Aperture Compressive Temporal Imaging)
section of the InverseNet paper. Contains ground truth, masks, measurements, and
reconstructed video frames for 3 scenarios x 4 methods x 28 measurement groups.

## Dataset Overview

| Item | Value |
|------|-------|
| Modality | CACTI (Snapshot Compressive Imaging) |
| Spatial resolution | 256 x 256 |
| Temporal frames | 8 per group |
| Videos | 6 (Kobe, Traffic, Runner, Drop, Crash, Aerial) |
| Measurement groups | 28 total |
| Methods | GAP-TV, PnP-FFDNet, ELP-Unfolding, EfficientSCI |
| Scenarios | 3 (Ideal, Baseline, Oracle) |
| Total arrays | 476 (17 per group x 28 groups) |
| Total size | ~608 MB (compressed .npz) |
| Data type | float32, range [0, 1] |

## Mismatch Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| mask_dx | 0.5 px | Horizontal sub-pixel shift |
| mask_dy | 0.3 px | Vertical sub-pixel shift |
| mask_theta | 0.1 deg | Rotation angle |
| mask_blur_sigma | 0.0 | No additional blur |
| gain | 1.02 | Radiometric gain mismatch |
| offset | 0.002 | Radiometric offset |
| noise_sigma | 1.0 | Gaussian noise std |
| noise_peak | 10000 | Poisson noise peak |

## Scenarios

- **Scenario I (Ideal):** Ideal measurement (from benchmark .mat) + ideal mask.
  Upper bound on reconstruction quality with no mismatch.

- **Scenario II (Baseline):** Corrupted measurement (warped mask + gain/offset + noise)
  + ideal mask (wrong operator). Shows degradation from operator mismatch.

- **Scenario III (Oracle):** Corrupted measurement with gain/offset corrected
  + true warped mask. Shows recovery when the true operator is known.

## Methods

| Method | Type | Reference |
|--------|------|-----------|
| GAP-TV | Optimization (GAP + Total Variation) | Yuan (2016) |
| PnP-FFDNet | Plug-and-Play (GAP + FFDNet denoiser) | Yuan et al. (2020) |
| ELP-Unfolding | Deep Unfolding (ADMM + learned priors) | ECCV 2022 |
| EfficientSCI | End-to-End (two-stage 3D CNN) | CVPR 2023 |

## File Structure

```
cacti_reconstructions/
    readme_cacti.md          # This file
    kobe.npz                 # 4 groups,  95 MB
    traffic.npz              # 6 groups, 130 MB
    runner.npz               # 5 groups, 109 MB
    drop.npz                 # 5 groups,  97 MB
    crash.npz                # 4 groups,  87 MB
    aerial.npz               # 4 groups,  90 MB

../cacti_summary.json              # Aggregated metrics (per-video, overall)
../cacti_validation_results.json   # Per-group detailed metrics (PSNR, SSIM)
```

## Array Keys per Group

Each `.npz` file contains arrays keyed as `g{i}_{name}` where `i` is the group
index (0-based) and `{name}` is one of:

| Key | Shape | Description |
|-----|-------|-------------|
| `gt` | (256, 256, 8) | Ground truth video frames |
| `mask_ideal` | (256, 256, 8) | Original binary coded-aperture mask |
| `mask_warped` | (256, 256, 8) | Binarized warped mask (shifted by dx, dy, theta) |
| `meas_ideal` | (256, 256) | Ideal measurement = sum(gt * mask_ideal, axis=2) |
| `meas_corrupt` | (256, 256) | Corrupted measurement with mismatch + noise |
| `scenario_i_gap_tv` | (256, 256, 8) | GAP-TV reconstruction, Scenario I |
| `scenario_i_pnp_ffdnet` | (256, 256, 8) | PnP-FFDNet reconstruction, Scenario I |
| `scenario_i_elp_unfolding` | (256, 256, 8) | ELP-Unfolding reconstruction, Scenario I |
| `scenario_i_efficientsci` | (256, 256, 8) | EfficientSCI reconstruction, Scenario I |
| `scenario_ii_gap_tv` | (256, 256, 8) | GAP-TV reconstruction, Scenario II |
| `scenario_ii_pnp_ffdnet` | (256, 256, 8) | PnP-FFDNet reconstruction, Scenario II |
| `scenario_ii_elp_unfolding` | (256, 256, 8) | ELP-Unfolding reconstruction, Scenario II |
| `scenario_ii_efficientsci` | (256, 256, 8) | EfficientSCI reconstruction, Scenario II |
| `scenario_iii_gap_tv` | (256, 256, 8) | GAP-TV reconstruction, Scenario III |
| `scenario_iii_pnp_ffdnet` | (256, 256, 8) | PnP-FFDNet reconstruction, Scenario III |
| `scenario_iii_elp_unfolding` | (256, 256, 8) | ELP-Unfolding reconstruction, Scenario III |
| `scenario_iii_efficientsci` | (256, 256, 8) | EfficientSCI reconstruction, Scenario III |

**17 arrays per group x 28 groups = 476 arrays total.**

## Groups per Video

| Video | Groups | Group indices |
|-------|--------|---------------|
| kobe | 4 | g0, g1, g2, g3 |
| traffic | 6 | g0, g1, g2, g3, g4, g5 |
| runner | 5 | g0, g1, g2, g3, g4 |
| drop | 5 | g0, g1, g2, g3, g4 |
| crash | 4 | g0, g1, g2, g3 |
| aerial | 4 | g0, g1, g2, g3 |

## Usage Examples

### Load and inspect data

```python
import numpy as np

# Load one video
d = np.load("cacti_reconstructions/kobe.npz")

# List all keys
print(sorted(d.keys()))

# Load group 0 ground truth and a reconstruction
gt = d["g0_gt"]                           # (256, 256, 8) float32
recon_i = d["g0_scenario_i_elp_unfolding"]  # (256, 256, 8) float32

# Access a single frame (e.g., frame 3)
frame_gt = gt[:, :, 3]       # (256, 256)
frame_recon = recon_i[:, :, 3]  # (256, 256)
```

### Compute PSNR for a single frame

```python
def psnr(x, y, max_val=1.0):
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(max_val ** 2 / mse)

gt = d["g0_gt"]
recon = d["g0_scenario_iii_elp_unfolding"]
print(f"PSNR: {psnr(recon, gt):.2f} dB")
```

### Visual comparison across scenarios (one frame)

```python
import matplotlib.pyplot as plt

d = np.load("cacti_reconstructions/kobe.npz")
frame = 3  # pick a frame
method = "elp_unfolding"

gt    = d["g0_gt"][:, :, frame]
sc_i  = d[f"g0_scenario_i_{method}"][:, :, frame]
sc_ii = d[f"g0_scenario_ii_{method}"][:, :, frame]
sc_iii= d[f"g0_scenario_iii_{method}"][:, :, frame]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, img, title in zip(axes,
    [gt, sc_i, sc_ii, sc_iii],
    ["Ground Truth", "Scenario I (Ideal)", "Scenario II (Baseline)", "Scenario III (Oracle)"]):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig("cacti_visual_comparison.png", dpi=150)
plt.show()
```

### Error map visualization

```python
gt = d["g0_gt"][:, :, 3]
recon = d["g0_scenario_ii_elp_unfolding"][:, :, 3]
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
plt.savefig("cacti_error_map.png", dpi=150)
```

### Compare all methods for one scenario

```python
d = np.load("cacti_reconstructions/runner.npz")
frame = 4
scenario = "scenario_i"
methods = ["gap_tv", "pnp_ffdnet", "elp_unfolding", "efficientsci"]
labels = ["GAP-TV", "PnP-FFDNet", "ELP-Unfolding", "EfficientSCI"]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(d["g0_gt"][:, :, frame], cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Ground Truth")
for ax, m, lab in zip(axes[1:], methods, labels):
    ax.imshow(d[f"g0_{scenario}_{m}"][:, :, frame], cmap="gray", vmin=0, vmax=1)
    ax.set_title(lab)
for ax in axes:
    ax.axis("off")
plt.suptitle(f"Scenario I — Runner, Frame {frame}")
plt.tight_layout()
plt.savefig("cacti_method_comparison.png", dpi=150)
```

### Load metrics from JSON

```python
import json

with open("../cacti_summary.json") as f:
    summary = json.load(f)

# Overall PSNR per method per scenario
for scenario in ["scenario_i", "scenario_ii", "scenario_iii"]:
    print(f"\n{scenario}:")
    for method, stats in summary["overall"][scenario].items():
        print(f"  {method}: {stats['psnr_mean']:.2f} +/- {stats['psnr_std']:.2f} dB")
```

### Mask comparison (ideal vs warped)

```python
d = np.load("cacti_reconstructions/kobe.npz")
mask_ideal = d["g0_mask_ideal"][:, :, 0]
mask_warped = d["g0_mask_warped"][:, :, 0]
diff = np.abs(mask_ideal.astype(float) - mask_warped.astype(float))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(mask_ideal, cmap="gray")
axes[0].set_title("Ideal Mask (frame 0)")
axes[1].imshow(mask_warped, cmap="gray")
axes[1].set_title("Warped Mask (frame 0)")
axes[2].imshow(diff, cmap="hot")
axes[2].set_title(f"Difference ({diff.sum():.0f} pixels changed)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("cacti_mask_comparison.png", dpi=150)
```

## Validation Results Summary

| Method | Scenario I (Ideal) | Scenario II (Baseline) | Scenario III (Oracle) | Gap I-II | Recovery II-III |
|--------|-------------------|----------------------|---------------------|----------|-----------------|
| GAP-TV | 26.75 +/- 4.48 | 15.81 +/- 1.98 | 26.01 +/- 3.72 | +10.94 | +10.21 |
| PnP-FFDNet | 29.28 +/- 5.53 | 11.43 +/- 2.71 | 25.39 +/- 3.52 | +17.85 | +13.96 |
| ELP-Unfolding | 34.09 +/- 4.11 | 15.47 +/- 1.71 | 29.40 +/- 3.15 | +18.63 | +13.93 |
| EfficientSCI | 35.39 +/- 4.46 | 14.81 +/- 2.19 | 27.38 +/- 3.52 | +20.58 | +12.57 |

All PSNR values in dB. Gap = degradation from mismatch. Recovery = improvement from oracle correction.

## Reproduction

To regenerate this dataset:

```bash
cd /home/spiritai/PWM/test2/Physics_World_Model
python papers/inversenet/scripts/validate_cacti_inversenet.py --device cuda --save-recon
```

Requires: PyTorch, scipy, numpy, and pretrained model checkpoints for ELP-Unfolding,
EfficientSCI, and FFDNet (see validate_cacti_inversenet.py for paths).

## Source Data

The SCI benchmark videos (Kobe, Traffic, Runner, Drop, Crash, Aerial) are loaded
from `.mat` files via `pwm_core.data.loaders.cacti_bench.CACTIBenchmark`.
Each video provides multiple 8-frame measurement groups with pre-computed
binary masks and ideal measurements.
