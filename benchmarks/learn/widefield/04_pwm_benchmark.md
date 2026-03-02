# 04 — PWM Benchmark: Widefield Fluorescence Microscopy

## 1. Overview

The PWM benchmark for **Widefield Fluorescence Microscopy** evaluates reconstruction algorithms
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
| Dataset ID | `biosr_factin` |
| Dataset URL | https://figshare.com/articles/dataset/BioSR/13744429 |
| Fallback | `generated` |
| Synthetic generator | `cell_phantom` |
| Citation | Qiao et al., Evaluation and development of deep neural networks for image super-resolution in optical microscopy, Nature Methods 2021 |
| License | CC-BY-4.0 |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| PSF sigma | 2.0 | 1.2 – 3.5 | px |
| Background level | 50.0 | 0.0 – 200.0 | counts |
| Gain | 1.0 | 0.85 – 1.15 | - |
| Flatfield non-uniformity | 0.0 | 0.0 – 15.0 | peak-to-peak |
| Photobleaching rate | 0.0 | 0.0 – 0.05 | per frame |


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
python benchmarks/runners/run_expanded.py --modality widefield

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality widefield --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/widefield.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/widefield_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
