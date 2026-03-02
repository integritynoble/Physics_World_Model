# 04 — PWM Benchmark: Panorama Multi-Focus Fusion

## 1. Overview

The PWM benchmark for **Panorama Multi-Focus Fusion** evaluates reconstruction algorithms
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
| Object (x) | [512, 512, 3] |
| Measurements (y) | [512, 512, 3] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `div2k_multicrop` |
| Dataset URL | https://data.vision.ee.ethz.ch/cvl/DIV2K/ |
| Fallback | `generated` |
| Synthetic generator | `cell_phantom` |
| Citation | Agustsson and Timofte, NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study, CVPRW 2017 |
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
python benchmarks/runners/run_expanded.py --modality panorama

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality panorama --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/panorama.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/panorama_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
