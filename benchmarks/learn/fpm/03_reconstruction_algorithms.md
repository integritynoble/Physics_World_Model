# 03 — Reconstruction Algorithms: Fourier Ptychographic Microscopy (FPM)

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Fourier Ptychographic Microscopy (FPM) is **`sequential_phase_retrieval`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | Sequential Phase Retrieval | `pwm_core.recon.fpm_solver` | `run_fpm` | No |  |
| best_quality | Gradient Descent FPM | `pwm_core.recon.fpm_solver` | `gradient_descent_fpm_recon` | No | Tian et al. 2014, Biomed. Optics Express |
| famous_dl | Fourier Ptychnet | `pwm_core.recon.fpm_solver` | `fourier_ptychnet_recon` | No | Jiang et al. 2018, Biomed. Optics Express |
| small_gpu | Fourier Ptychnet | `pwm_core.recon.fpm_solver` | `fourier_ptychnet_recon` | No |  |


---

## 3. Solver Details

### Traditional Cpu: Sequential Phase Retrieval

- **Module**: `pwm_core.recon.fpm_solver`
- **Function**: `run_fpm`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: Gradient Descent FPM

- **Module**: `pwm_core.recon.fpm_solver`
- **Function**: `gradient_descent_fpm_recon`
- **Parameters**: 0
- **GPU required**: No
- **Reference**: Tian et al. 2014, Biomed. Optics Express

### Famous Dl: Fourier Ptychnet

- **Module**: `pwm_core.recon.fpm_solver`
- **Function**: `fourier_ptychnet_recon`
- **Parameters**: 7M
- **GPU required**: No
- **Reference**: Jiang et al. 2018, Biomed. Optics Express

### Small Gpu: Fourier Ptychnet

- **Module**: `pwm_core.recon.fpm_solver`
- **Function**: `fourier_ptychnet_recon`
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
3. Add the solver tier to the modality config in `benchmarks/configs/fpm.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
