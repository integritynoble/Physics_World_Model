# 04 — PWM Benchmark: Generic Matrix Sensing

## 1. Overview

The PWM benchmark for **Generic Matrix Sensing** evaluates reconstruction algorithms
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
| Measurements (y) | [614] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `shepp_logan_synth` |
| Dataset URL | https://github.com/pwm-project/shepp-logan-synth |
| Fallback | `generated` |
| Synthetic generator | `spectral_scene` |
| Citation | Shepp and Logan, The Fourier reconstruction of a head section, IEEE TNS 1974 |
| License | Public domain |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Matrix perturbation | 0.0 | 0.0 – 10.0 | A |
| Condition number change | 0.0 | 0 – 0 | - |


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
python benchmarks/runners/run_expanded.py --modality matrix

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality matrix --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/matrix.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/matrix_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
