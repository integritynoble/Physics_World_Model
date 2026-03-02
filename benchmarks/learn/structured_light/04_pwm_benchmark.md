# 04 — PWM Benchmark: Structured-Light Depth Camera

## 1. Overview

The PWM benchmark for **Structured-Light Depth Camera** evaluates reconstruction algorithms
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
| Dataset ID | `middlebury_sl` |
| Dataset URL | https://vision.middlebury.edu/stereo/data/ |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | Scharstein and Szeliski, A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms, IJCV 2002 |
| License | Research use |

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
python benchmarks/runners/run_expanded.py --modality structured_light

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality structured_light --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/structured_light.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/structured_light_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
