# 04 — PWM Benchmark: Diffuse Optical Tomography (DOT)

## 1. Overview

The PWM benchmark for **Diffuse Optical Tomography (DOT)** evaluates reconstruction algorithms
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
| Object (x) | [64, 64, 64] |
| Measurements (y) | [256] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `toast_phantom` |
| Dataset URL | http://web4.cs.ucl.ac.uk/research/vis/toast/download.html |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | Schweiger and Arridge, The Toast++ software suite for forward and inverse modeling in optical tomography, J. Biomed. Optics 2014 |
| License | LGPL-2.1 |

---

## 4. Mismatch Parameters

No mismatch parameters defined for this modality.


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
python benchmarks/runners/run_expanded.py --modality dot

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality dot --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/dot.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/dot_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
