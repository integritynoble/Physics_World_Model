# 04 — PWM Benchmark: Cryo-EM Single Particle Analysis

## 1. Overview

The PWM benchmark for **Cryo-EM Single Particle Analysis** evaluates reconstruction algorithms
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
| Dataset ID | `cryo_em` |
| Dataset URL | https://www.ebi.ac.uk/empiar/EMPIAR-10028/ |
| Fallback | `generated` |
| Synthetic generator | `cell_phantom` |
| Citation | Bartesaghi et al., Science 2015 |
| License | CC-BY-4.0 |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Defocus error | 0.0 | -500.0 – 500.0 | nm |
| Astigmatism | 0.0 | 0.0 – 100.0 | nm |
| Beam tilt | 0.0 | -1.0 – 1.0 | mrad |
| Ice thickness variation | 50.0 | 30.0 – 80.0 | nm |


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
python benchmarks/runners/run_expanded.py --modality cryo_em

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality cryo_em --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/cryo_em.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/cryo_em_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
