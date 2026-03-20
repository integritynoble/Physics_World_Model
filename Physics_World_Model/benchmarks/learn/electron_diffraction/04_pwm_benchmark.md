# 04 — PWM Benchmark: 4D-STEM Electron Diffraction

## 1. Overview

The PWM benchmark for **4D-STEM Electron Diffraction** evaluates reconstruction algorithms
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
| Object (x) | [128, 128] |
| Measurements (y) | [128, 128] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `py4dstem_tutorial` |
| Dataset URL | https://github.com/py4dstem/py4DSTEM_tutorials |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | Savitzky et al., py4DSTEM: A Software Package for Four-Dimensional Scanning Transmission Electron Microscopy, Microscopy and Microanalysis 2021 |
| License | GPL-3.0 |

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
python benchmarks/runners/run_expanded.py --modality electron_diffraction

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality electron_diffraction --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/electron_diffraction.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/electron_diffraction_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
