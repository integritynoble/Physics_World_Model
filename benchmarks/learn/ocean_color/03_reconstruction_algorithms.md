# 03 — Reconstruction Algorithms: Ocean Color Remote Sensing

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Ocean Color Remote Sensing is **`atmospheric_correction`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | RDA | `pwm_core.recon.sar_rda` | `range_doppler` | No |  |
| best_quality | SAR-DL | `pwm_core.recon.sar_dl` | `sar_dl_recon` | Yes |  |


---

## 3. Solver Details

### Traditional Cpu: RDA

- **Module**: `pwm_core.recon.sar_rda`
- **Function**: `range_doppler`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: SAR-DL

- **Module**: `pwm_core.recon.sar_dl`
- **Function**: `sar_dl_recon`
- **Parameters**: 5M
- **GPU required**: Yes


---

## 4. Algorithm Selection Guide

| Scenario | Recommended Tier | Why |
|----------|------------------|-----|
| Quick baseline | `traditional_cpu` | Fast, no GPU needed |
| Best quality | `best_quality` | Highest PSNR/SSIM |
| Published benchmark | `famous_dl` | Reproducible, citable |
| Limited GPU memory | `small_gpu` | Fits on consumer GPU |

### General Recommendations

- Start with `traditional_cpu` to establish a baseline
- Compare with `best_quality` to see the achievable improvement
- Use `famous_dl` if you need results comparable to published papers
- Use `small_gpu` if GPU memory is limited (< 6 GB VRAM)

---

## 5. Adding a New Solver

To add a new solver to the benchmark:

1. Implement the solver function in `packages/pwm_core/pwm_core/recon/`
2. Register it in `packages/pwm_core/contrib/solver_registry.yaml`
3. Add the solver tier to the modality config in `benchmarks/configs/ocean_color.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
