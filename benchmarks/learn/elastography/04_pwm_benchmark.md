# 04 — PWM Benchmark: Shear-Wave Elastography

## 1. Overview

The PWM benchmark for **Shear-Wave Elastography** evaluates reconstruction algorithms
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
| Measurements (y) | [128, 512] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `cirs_phantom_synth` |
| Dataset URL | https://github.com/pwm-project/cirs-phantom-synth |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | PWM Project, Synthetic elastography phantom generator, 2026 |
| License | MIT |

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
python benchmarks/runners/run_expanded.py --modality elastography

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality elastography --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/elastography.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/elastography_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
