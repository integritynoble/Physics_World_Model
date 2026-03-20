# 03 — Reconstruction Algorithms: Neural Radiance Fields (NeRF)

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Neural Radiance Fields (NeRF) is **`nerf_mlp`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | SfM + MVS | `pwm_core.recon.nerf_solver` | `run_nerf` | No |  |
| best_quality | Mip-NeRF 360 | `pwm_core.recon.nerf_solver` | `run_nerf` | Yes | Barron et al. CVPR 2022 |
| famous_dl | NeRF (original MLP) | `pwm_core.recon.nerf_solver` | `run_nerf` | No | Mildenhall et al. 2020 |
| small_gpu | Instant-NGP | `pwm_core.recon.nerf_solver` | `run_nerf` | No | Muller et al. 2022 |


---

## 3. Solver Details

### Traditional Cpu: SfM + MVS

- **Module**: `pwm_core.recon.nerf_solver`
- **Function**: `run_nerf`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: Mip-NeRF 360

- **Module**: `pwm_core.recon.nerf_solver`
- **Function**: `run_nerf`
- **Parameters**: 9M
- **GPU required**: Yes
- **Reference**: Barron et al. CVPR 2022

### Famous Dl: NeRF (original MLP)

- **Module**: `pwm_core.recon.nerf_solver`
- **Function**: `run_nerf`
- **Parameters**: 1.2M
- **GPU required**: No
- **Reference**: Mildenhall et al. 2020

### Small Gpu: Instant-NGP

- **Module**: `pwm_core.recon.nerf_solver`
- **Function**: `run_nerf`
- **Parameters**: 5M
- **GPU required**: No
- **Reference**: Muller et al. 2022


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
3. Add the solver tier to the modality config in `benchmarks/configs/nerf.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
