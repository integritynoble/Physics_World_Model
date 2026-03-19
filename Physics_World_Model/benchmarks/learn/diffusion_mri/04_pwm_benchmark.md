# 04 — PWM Benchmark: Diffusion MRI (DTI)

## 1. Overview

The PWM benchmark for **Diffusion MRI (DTI)** evaluates reconstruction algorithms
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
| Dataset ID | `hcp_diffusion` |
| Dataset URL | https://db.humanconnectome.org/ |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | Van Essen et al., The WU-Minn Human Connectome Project: An overview, NeuroImage 2013 |
| License | HCP Open Access Data Use Terms |

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
python benchmarks/runners/run_expanded.py --modality diffusion_mri

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality diffusion_mri --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/diffusion_mri.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/diffusion_mri_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
