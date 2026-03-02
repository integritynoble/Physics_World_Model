# 04 — PWM Benchmark: Coded Aperture Snapshot Spectral Imaging (CASSI)

## 1. Overview

The PWM benchmark for **Coded Aperture Snapshot Spectral Imaging (CASSI)** evaluates reconstruction algorithms
under physics model mismatch using a 3-tier structure with increasing
difficulty.

---

## 2. Three-Tier Structure

| Tier | Mismatch | Purpose |
|------|----------|---------|
| **Public** | Mild | Algorithm development, debugging |
| **Dev** | Moderate | Validation, hyperparameter tuning |
| **Hidden** | Severe | Final evaluation, leaderboard |

---

## 3. Data Format

### Signal Dimensions

| Dimension | Shape |
|-----------|-------|
| Object (x) | [256, 256, 28] |
| Measurements (y) | [256, 310] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `kaist_mst_10scenes` |
| Dataset URL | https://github.com/caiyuanhao1998/MST/tree/main/simulation/test_datasets |
| Fallback | `generated` |
| Synthetic generator | `spectral_scene` |
| Citation | Cai et al., Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction, CVPR 2022 |
| License | MIT |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Mask shift dx | 0.0 | -3.0 – 3.0 | px |
| Mask shift dy | 0.0 | -3.0 – 3.0 | px |
| Mask rotation | 0.0 | -2.0 – 2.0 | deg |
| Dispersion slope a1 | 2.0 | 1.5 – 2.5 | px/band |
| Dispersion offset alpha | 0.0 | -0.5 – 0.5 | px |
| Gain | 1.0 | 0.9 – 1.1 | - |
| Read noise | 5.0 | 1.0 – 15.0 | e- |


Each sample in the benchmark has randomly drawn mismatch values from the
ranges above. The mismatch severity increases from public to hidden tier.

---

## 5. Scoring

### Metrics

| Metric | Primary | Threshold |
|--------|:-------:|-----------|
| psnr | Yes | — |
| ssim | No | — |
| sam | No | — |

Metrics are computed using `benchmarks/framework/metrics.py`:

```python
from benchmarks.framework.metrics import compute_psnr, compute_ssim

psnr = compute_psnr(x_true, x_hat, max_val=1.0)
ssim = compute_ssim(x_true, x_hat, data_range=1.0)
```

---

## 6. Running the Benchmark

```bash
# Using the expanded config runner
python benchmarks/runners/run_expanded.py --modality cassi

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality cassi --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/cassi.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/cassi_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
