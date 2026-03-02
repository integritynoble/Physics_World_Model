# 04 — PWM Benchmark: Light-Sheet Fluorescence Microscopy (LSFM)

## 1. Overview

The PWM benchmark for **Light-Sheet Fluorescence Microscopy (LSFM)** evaluates reconstruction algorithms
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
| Object (x) | [512, 512, 128] |
| Measurements (y) | [512, 512, 128] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `openspim_demo` |
| Dataset URL | https://openspim.org/downloads |
| Fallback | `generated` |
| Synthetic generator | `cell_phantom` |
| Citation | Pitrone et al., OpenSPIM: an open-access light-sheet microscopy platform, Nature Methods 2013 |
| License | CC-BY-SA-4.0 |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Sheet thickness | 5.0 | 2.0 – 15.0 | um |
| Sheet tilt | 0.0 | -3.0 – 3.0 | deg |
| Stripe strength | 0.2 | 0.0 – 0.8 | relative |
| Attenuation coeff | 0.02 | 0.005 – 0.08 | per slice |


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
python benchmarks/runners/run_expanded.py --modality lightsheet

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality lightsheet --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/lightsheet.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/lightsheet_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
