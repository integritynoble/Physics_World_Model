# 03 — Reconstruction Algorithms: Light Field Imaging

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Light Field Imaging is **`shift_and_sum`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | Shift-and-Sum | `pwm_core.recon.light_field_solver` | `run_light_field` | No |  |
| best_quality | LFBM5D | `pwm_core.recon.light_field_solver` | `lfbm5d_recon` | No | Alain et al. 2017, Signal Processing: Image Communication |
| famous_dl | LFSSR | `pwm_core.recon.light_field_solver` | `lfssr_recon` | No | Yeung et al. ECCV 2018 |
| small_gpu | LFSSR | `pwm_core.recon.light_field_solver` | `lfssr_recon` | No |  |


---

## 3. Solver Details

### Traditional Cpu: Shift-and-Sum

- **Module**: `pwm_core.recon.light_field_solver`
- **Function**: `run_light_field`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: LFBM5D

- **Module**: `pwm_core.recon.light_field_solver`
- **Function**: `lfbm5d_recon`
- **Parameters**: 0
- **GPU required**: No
- **Reference**: Alain et al. 2017, Signal Processing: Image Communication

### Famous Dl: LFSSR

- **Module**: `pwm_core.recon.light_field_solver`
- **Function**: `lfssr_recon`
- **Parameters**: 1.5M
- **GPU required**: No
- **Reference**: Yeung et al. ECCV 2018

### Small Gpu: LFSSR

- **Module**: `pwm_core.recon.light_field_solver`
- **Function**: `lfssr_recon`
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
3. Add the solver tier to the modality config in `benchmarks/configs/light_field.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
