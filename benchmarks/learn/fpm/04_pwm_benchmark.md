# 04 — PWM Benchmark: Fourier Ptychographic Microscopy (FPM)

## 1. Overview

The PWM benchmark for **Fourier Ptychographic Microscopy (FPM)** evaluates reconstruction algorithms
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
| Object (x) | [1024, 1024] |
| Measurements (y) | [256, 256, 225] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `fpm_led_benchmark` |
| Dataset URL | https://github.com/zhenglab/FPM/tree/master/data |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | Zheng et al., Wide-field, high-resolution Fourier ptychographic microscopy, Nature Photonics 2013 |
| License | BSD-3-Clause |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| LED position error | 0.0 | 0 – 0 | mm |
| LED intensity variation | 1.0 | 0.5 – 1.5 | relative |
| Pupil aberration (Zernike) | 0.0 | 0.0 – 0.3 | waves |
| Defocus | 0.0 | -5.0 – 5.0 | um |


Each sample in the benchmark has randomly drawn mismatch values from the
ranges above. The mismatch severity increases from public to hidden tier.

---

## 5. Scoring

### Metrics

| Metric | Primary | Threshold |
|--------|:-------:|-----------|
| psnr | Yes | — |
| ssim | No | — |

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
python benchmarks/runners/run_expanded.py --modality fpm

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality fpm --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/fpm.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/fpm_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
