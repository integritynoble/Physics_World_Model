# 04 — PWM Benchmark: Confocal 3D Z-Stack

## 1. Overview

The PWM benchmark for **Confocal 3D Z-Stack** evaluates reconstruction algorithms
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
| Object (x) | [256, 256, 64] |
| Measurements (y) | [256, 256, 64] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `care_tribolium` |
| Dataset URL | https://publications.mpi-cbg.de/publications-sites/7207/ |
| Fallback | `generated` |
| Synthetic generator | `cell_phantom` |
| Citation | Weigert et al., Content-aware image restoration: pushing the limits of fluorescence microscopy, Nature Methods 2018 |
| License | CC-BY-4.0 |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Axial PSF sigma | 3.0 | 1.5 – 6.0 | px |
| Refractive index | 1.515 | 1.33 – 1.56 | - |
| Attenuation coeff | 0.03 | 0.0 – 0.1 | per slice |
| Spherical aberration | 0.0 | 0.0 – 0.5 | waves |


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
python benchmarks/runners/run_expanded.py --modality confocal_3d

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality confocal_3d --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/confocal_3d.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/confocal_3d_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
