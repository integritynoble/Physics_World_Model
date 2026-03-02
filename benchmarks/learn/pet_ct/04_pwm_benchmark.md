# 04 — PWM Benchmark: PET/CT Fusion

## 1. Overview

The PWM benchmark for **PET/CT Fusion** evaluates reconstruction algorithms
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
| Dataset ID | `pet_dataset` |
| Dataset URL | https://ultra-low-dose-pet.grand-challenge.org/ |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | Ultra-low-dose PET Challenge 2022 |
| License | Research use |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| CT-PET registration error | 0.0 | 0.0 – 3.0 | mm |
| Attenuation map from CT error | 0.0 | 0.0 – 10.0 | HU-to-LAC |
| Respiratory motion mismatch | 0.0 | 0.0 – 15.0 | mm |
| CT contrast agent artifact | 0.0 | 0.0 – 20.0 | attenuation |


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
python benchmarks/runners/run_expanded.py --modality pet_ct

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality pet_ct --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/pet_ct.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/pet_ct_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
