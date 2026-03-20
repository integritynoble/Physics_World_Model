# 04 — PWM Benchmark: Ultrasound B-mode Imaging

## 1. Overview

The PWM benchmark for **Ultrasound B-mode Imaging** evaluates reconstruction algorithms
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
| Dataset ID | `picmus_challenge` |
| Dataset URL | https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/download.html |
| Fallback | `generated` |
| Synthetic generator | `shepp_logan` |
| Citation | Liebgott et al., Plane-Wave Imaging Challenge in Medical Ultrasound, IEEE IUS 2016 |
| License | Research use |

---

## 4. Mismatch Parameters

| Parameter | Nominal | Range | Unit |
|-----------|---------|-------|------|
| Speed of sound | 1540.0 | 1450.0 – 1600.0 | m/s |
| Phase aberration | 0.0 | 0.0 – 50.0 | ns rms |
| Element sensitivity | 1.0 | 0.7 – 1.3 | - |
| Attenuation | 0.5 | 0.3 – 0.8 | dB/cm/MHz |


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
python benchmarks/runners/run_expanded.py --modality ultrasound

# Quick test with specific solver
python benchmarks/runners/run_expanded.py --modality ultrasound --solver traditional_cpu
```

---

## 7. Configuration File

The full benchmark configuration is at:
```
benchmarks/configs/ultrasound.yaml
```

The expanded configuration (with data sources and full parameters) is at:
```
benchmarks/expanded_configs/ultrasound_expanded.yaml
```

---

*Previous: [03 — Reconstruction Algorithms](03_reconstruction_algorithms.md)*
*Next: [05 — Hands-On Tutorial](05_hands_on_tutorial.md)*
