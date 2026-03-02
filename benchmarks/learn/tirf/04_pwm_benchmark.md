# 04 — PWM Benchmark: TIRF Microscopy

## 1. Overview

The PWM benchmark for **TIRF Microscopy** evaluates reconstruction algorithms
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
| Dataset ID | `tirf_sim_benchmark` |
| Dataset URL | https://github.com/pwm-project/tirf-sim-benchmark |
| Fallback | `generated` |
| Synthetic generator | `cell_phantom` |
| Citation | PWM Project, TIRF-SIM synthetic benchmark dataset, 2026 |
| License | MIT |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Incidence angle | 68.0 | 62.0 – 75.0 | deg |
| Evanescent depth | 100.0 | 50.0 – 300.0 | nm |
| Background (non-TIRF) | 0.0 | 0.0 – 0.3 | relative |


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
python benchmarks/runners/run_expanded.py --modality tirf

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality tirf --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/tirf.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/tirf_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
