# 04 — PWM Benchmark: X-ray Fluorescence (XRF) Imaging

## 1. Overview

The PWM benchmark for **X-ray Fluorescence (XRF) Imaging** evaluates reconstruction algorithms
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
| Dataset ID | `xrf_dataset` |
| Dataset URL | https://www.aps.anl.gov/Science/Scientific-Software/XRF-Maps |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | Vogt et al., APS XRF Maps |
| License | Public domain |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Excitation energy drift | 0.0 | -0.05 – 0.05 | keV |
| Detector resolution | 130.0 | 120.0 – 150.0 | eV |
| Matrix absorption | 1.0 | 0.85 – 1.15 | - |
| Beam spot size | 1.0 | 0.5 – 2.0 | um |


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
python benchmarks/runners/run_expanded.py --modality xrf_imaging

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality xrf_imaging --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/xrf_imaging.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/xrf_imaging_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
