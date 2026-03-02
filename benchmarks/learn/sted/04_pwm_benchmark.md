# 04 — PWM Benchmark: STED Microscopy

## 1. Overview

The PWM benchmark for **STED Microscopy** evaluates reconstruction algorithms
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
| Object (x) | [512, 512] |
| Measurements (y) | [512, 512] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `abberior_sted` |
| Dataset URL | https://abberior.com/sted-microscopy/ |
| Fallback | `generated` |
| Synthetic generator | `cell_phantom` |
| Citation | Hell and Wichmann, Breaking the diffraction resolution limit by stimulated emission, Optics Letters 1994 |
| License | Research use |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Depletion beam alignment | 0.0 | 0.0 – 30.0 | nm offset |
| Saturation factor | 30.0 | 10.0 – 50.0 | - |
| Effective PSF FWHM | 40.0 | 30.0 – 120.0 | nm |


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
python benchmarks/runners/run_expanded.py --modality sted

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality sted --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/sted.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/sted_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
