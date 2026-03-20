# 04 — PWM Benchmark: Confocal Live-Cell Microscopy

## 1. Overview

The PWM benchmark for **Confocal Live-Cell Microscopy** evaluates reconstruction algorithms
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
| Dataset ID | `deepbacs_fluor` |
| Dataset URL | https://zenodo.org/record/5764540 |
| Fallback | `generated` |
| Synthetic generator | `cell_phantom` |
| Citation | Spahn et al., DeepBacs for multi-task bacterial image analysis using open-source deep learning approaches, Communications Biology 2022 |
| License | CC-BY-4.0 |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| PSF sigma | 1.5 | 0.8 – 3.0 | px |
| Drift rate | 0.1 | 0.0 – 1.0 | px/frame |
| Bleaching rate | 0.01 | 0.0 – 0.1 | per frame |
| Pinhole misalignment | 0.0 | 0.0 – 0.5 | AU offset |


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
python benchmarks/runners/run_expanded.py --modality confocal_livecell

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality confocal_livecell --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/confocal_livecell.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/confocal_livecell_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
