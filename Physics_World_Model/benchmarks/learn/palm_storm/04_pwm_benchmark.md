# 04 — PWM Benchmark: PALM/STORM Single-Molecule Localization

## 1. Overview

The PWM benchmark for **PALM/STORM Single-Molecule Localization** evaluates reconstruction algorithms
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
| Dataset ID | `smlm_challenge_2016` |
| Dataset URL | http://bigwww.epfl.ch/smlm/challenge2016/datasets/ |
| Fallback | `generated` |
| Synthetic generator | `spectral_scene` |
| Citation | Sage et al., Super-resolution fight club: assessment of 2D and 3D single-molecule localization microscopy software, Nature Methods 2019 |
| License | CC-BY-4.0 |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Drift rate (x, y) | 0.0 | 0.0 – 2.0 | nm/frame |
| Background photons | 20.0 | 5.0 – 100.0 | per px |
| Photon count/event | 1000.0 | 200.0 – 5000.0 | photons |
| Pixel size | 100.0 | 90.0 – 110.0 | nm |


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
python benchmarks/runners/run_expanded.py --modality palm_storm

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality palm_storm --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/palm_storm.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/palm_storm_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
