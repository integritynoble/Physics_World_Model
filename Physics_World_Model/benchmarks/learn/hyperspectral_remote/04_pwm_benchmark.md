# 04 — PWM Benchmark: Hyperspectral Remote Sensing

## 1. Overview

The PWM benchmark for **Hyperspectral Remote Sensing** evaluates reconstruction algorithms
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
| Object (x) | [64, 64] |
| Measurements (y) | [64, 64] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `sentinel` |
| Dataset URL | https://scihub.copernicus.eu/ |
| Fallback | `generated` |
| Synthetic generator | `spectral_scene` |
| Citation | Copernicus Sentinel-2 |
| License | CC-BY-SA-3.0 |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Spectral shift | 0.0 | -2.0 – 2.0 | nm |
| Smile distortion | 0.0 | -1.0 – 1.0 | px |
| Keystone distortion | 0.0 | -0.5 – 0.5 | px |
| Radiometric gain | 1.0 | 0.9 – 1.1 | - |


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
python benchmarks/runners/run_expanded.py --modality hyperspectral_remote

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality hyperspectral_remote --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/hyperspectral_remote.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/hyperspectral_remote_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
