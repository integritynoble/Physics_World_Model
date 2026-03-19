# 03 — Reconstruction Algorithms: Integral Photography

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Integral Photography is **`depth_estimation`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | Depth Estimation | `pwm_core.recon.integral_solver` | `run_integral` | No |  |
| best_quality | DIBR | `pwm_core.recon.integral_solver` | `dibr_recon` | No | Fehn 2004, Proc. SPIE |
| famous_dl | EPINet | `pwm_core.recon.integral_solver` | `epinet_recon` | No | Shin et al. CVPR 2018 |
| small_gpu | EPINet | `pwm_core.recon.integral_solver` | `epinet_recon` | No |  |


---

## 3. Solver Details

### Traditional Cpu: Depth Estimation

- **Module**: `pwm_core.recon.integral_solver`
- **Function**: `run_integral`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: DIBR

- **Module**: `pwm_core.recon.integral_solver`
- **Function**: `dibr_recon`
- **Parameters**: 0
- **GPU required**: No
- **Reference**: Fehn 2004, Proc. SPIE

### Famous Dl: EPINet

- **Module**: `pwm_core.recon.integral_solver`
- **Function**: `epinet_recon`
- **Parameters**: 5M
- **GPU required**: No
- **Reference**: Shin et al. CVPR 2018

### Small Gpu: EPINet

- **Module**: `pwm_core.recon.integral_solver`
- **Function**: `epinet_recon`
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
3. Add the solver tier to the modality config in `benchmarks/configs/integral.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
