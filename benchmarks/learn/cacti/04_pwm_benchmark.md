# 04 — PWM Benchmark: Coded Aperture Compressive Temporal Imaging (CACTI)

## 1. Overview

The PWM benchmark for **Coded Aperture Compressive Temporal Imaging (CACTI)** evaluates reconstruction algorithms
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
| Object (x) | [256, 256, 8] |
| Measurements (y) | [256, 256] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `sci_6gray` |
| Dataset URL | https://github.com/liuyang12/SCI/tree/master/data |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | Liu et al., Rank Minimization for Snapshot Compressive Imaging, IEEE TPAMI 2019 |
| License | BSD-3-Clause |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Spatial shift x,y | 0.0 | -3.0 – 3.0 | px |
| Rotation | 0.0 | -2.0 – 2.0 | deg |
| Temporal clock error | 0.0 | -0.5 – 0.5 | frame frac |
| Gain / offset | 0.0 | 0.9 – 1.1 | - / counts |
| Frame-dependent gain | 1.0 | 0.9 – 1.1 | - |


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
python benchmarks/runners/run_expanded.py --modality cacti

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality cacti --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/cacti.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/cacti_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
