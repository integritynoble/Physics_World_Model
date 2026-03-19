# 04 — PWM Benchmark: Fluorescence Lifetime Imaging (FLIM)

## 1. Overview

The PWM benchmark for **Fluorescence Lifetime Imaging (FLIM)** evaluates reconstruction algorithms
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
| Object (x) | [256, 256, 2] |
| Measurements (y) | [256, 256, 256] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `flim_fret_benchmark` |
| Dataset URL | https://zenodo.org/record/8139025 |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | Bhatt et al., FLIM-FRET benchmark dataset for fluorescence lifetime imaging, Scientific Data 2023 |
| License | CC-BY-4.0 |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| IRF width | 80.0 | 40.0 – 200.0 | ps |
| IRF shift | 0.0 | -50.0 – 50.0 | ps |
| Afterpulsing | 0.01 | 0.0 – 0.1 | relative |
| Pile-up fraction | 0.0 | 0.0 – 0.05 | - |


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
python benchmarks/runners/run_expanded.py --modality flim

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality flim --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/flim.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/flim_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
