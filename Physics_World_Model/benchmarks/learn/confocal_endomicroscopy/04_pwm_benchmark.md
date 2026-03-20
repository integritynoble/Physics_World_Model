# 04 — PWM Benchmark: Confocal Laser Endomicroscopy (CLE)

## 1. Overview

The PWM benchmark for **Confocal Laser Endomicroscopy (CLE)** evaluates reconstruction algorithms
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
| Dataset ID | `oct_retinal` |
| Dataset URL | https://www.kaggle.com/datasets/paultimothymooney/kermany2018 |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | Kermany et al., Cell 2018 |
| License | CC-BY-4.0 |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Fiber bundle honeycomb pattern | 0.0 | 0 – 0 | - |
| Motion artifact | 0.0 | 0.0 – 10.0 | px/frame |
| Fluorescein concentration variation | 1.0 | 0.3 – 3.0 | relative |


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
python benchmarks/runners/run_expanded.py --modality confocal_endomicroscopy

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality confocal_endomicroscopy --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/confocal_endomicroscopy.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/confocal_endomicroscopy_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
