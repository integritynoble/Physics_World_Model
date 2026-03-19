# 03 — Reconstruction Algorithms: Generic Matrix Sensing

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Generic Matrix Sensing is **`fista_l2`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | Tikhonov / FISTA-L2 | `pwm_core.recon.classical` | `run_fista_l2` | No |  |
| best_quality | Diffusion Posterior Sampling | `pwm_core.recon.diffusion_posterior` | `diffusion_posterior_sample` | Yes | Song et al. 2023 |
| famous_dl | LISTA | `pwm_core.recon.lista` | `lista_reconstruct` | No | Gregor & LeCun, ICML 2010 |
| small_gpu | LISTA | `pwm_core.recon.lista` | `lista_reconstruct` | No |  |


---

## 3. Solver Details

### Traditional Cpu: Tikhonov / FISTA-L2

- **Module**: `pwm_core.recon.classical`
- **Function**: `run_fista_l2`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: Diffusion Posterior Sampling

- **Module**: `pwm_core.recon.diffusion_posterior`
- **Function**: `diffusion_posterior_sample`
- **Parameters**: 60M
- **GPU required**: Yes
- **Reference**: Song et al. 2023

### Famous Dl: LISTA

- **Module**: `pwm_core.recon.lista`
- **Function**: `lista_reconstruct`
- **Parameters**: 0.5M
- **GPU required**: No
- **Reference**: Gregor & LeCun, ICML 2010

### Small Gpu: LISTA

- **Module**: `pwm_core.recon.lista`
- **Function**: `lista_reconstruct`
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
3. Add the solver tier to the modality config in `benchmarks/configs/matrix.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
