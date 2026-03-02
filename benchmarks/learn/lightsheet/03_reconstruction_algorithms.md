# 03 — Reconstruction Algorithms: Light-Sheet Fluorescence Microscopy (LSFM)

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Light-Sheet Fluorescence Microscopy (LSFM) is **`fourier_notch_destripe`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | Fourier Notch Filter | `pwm_core.recon.lightsheet_solver` | `run_lightsheet` | No |  |
| best_quality | VSNR | `pwm_core.recon.lightsheet_solver` | `vsnr_destripe` | No |  |
| famous_dl | DeStripe | `pwm_core.recon.destripe_net` | `destripe_denoise` | No | Liang et al. 2022 |
| small_gpu | DeStripe | `pwm_core.recon.destripe_net` | `destripe_denoise` | No |  |


---

## 3. Solver Details

### Traditional Cpu: Fourier Notch Filter

- **Module**: `pwm_core.recon.lightsheet_solver`
- **Function**: `run_lightsheet`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: VSNR

- **Module**: `pwm_core.recon.lightsheet_solver`
- **Function**: `vsnr_destripe`
- **Parameters**: 0
- **GPU required**: No

### Famous Dl: DeStripe

- **Module**: `pwm_core.recon.destripe_net`
- **Function**: `destripe_denoise`
- **Parameters**: 2M
- **GPU required**: No
- **Reference**: Liang et al. 2022

### Small Gpu: DeStripe

- **Module**: `pwm_core.recon.destripe_net`
- **Function**: `destripe_denoise`
- **Parameters**: 0
- **GPU required**: No


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
3. Add the solver tier to the modality config in `benchmarks/configs/lightsheet.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
