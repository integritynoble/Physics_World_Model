# 04 — PWM Benchmark: Structured Illumination Microscopy (SIM)

## 1. Overview

The PWM benchmark for **Structured Illumination Microscopy (SIM)** evaluates reconstruction algorithms
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
| Measurements (y) | [512, 512, 9] |

### Data Source

| Property | Value |
|----------|-------|
| Dataset ID | `biosr_sim` |
| Dataset URL | https://figshare.com/articles/dataset/BioSR/13744429 |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | Qiao et al., Evaluation and development of deep neural networks for image super-resolution in optical microscopy, Nature Methods 2021 |
| License | CC-BY-4.0 |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Pattern frequency | 0.1 | 0.05 – 0.15 | cycles/px |
| Phase shifts | 0.0 | 0 – 0 | rad |
| Modulation depth | 0.8 | 0.3 – 1.0 | - |
| Pattern orientation | 0.0 | 0 – 0 | deg |


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
python benchmarks/runners/run_expanded.py --modality sim

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality sim --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/sim.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/sim_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
