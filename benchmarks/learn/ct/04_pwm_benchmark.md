# 04 — PWM Benchmark: X-ray Computed Tomography (CT)

## 1. Overview

The PWM benchmark for **X-ray Computed Tomography (CT)** evaluates reconstruction algorithms
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
| Object (x) | [256, 256] |
| Measurements (y) | [180, 256] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `lodopab_ct` |
| Dataset URL | https://zenodo.org/record/3384092 |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | Leuschner et al., LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction, Scientific Data 2021 |
| License | CC-BY-4.0 |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Center-of-rotation offset | 0.0 | -5.0 – 5.0 | px |
| Angular offset | 0.0 | -3.0 – 3.0 | deg |
| Detector tilt | 0.0 | -2.0 – 2.0 | deg |
| Beam hardening coeff | 0.0 | 0.0 – 0.05 | - |
| Ring artifact amplitude | 0.0 | 0.0 – 50.0 | counts |


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
python benchmarks/runners/run_expanded.py --modality ct

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality ct --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/ct.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/ct_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
