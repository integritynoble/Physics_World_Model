# 03 — Reconstruction Algorithms: Coherent Anti-Stokes Raman (CARS) Microscopy

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Coherent Anti-Stokes Raman (CARS) Microscopy is **`nrb_removal`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | Adjoint | `pwm_core.recon.adjoint` | `run_adjoint` | No |  |
| best_quality | PnP-ADMM | `pwm_core.recon.pnp_admm` | `pnp_admm_recon` | Yes |  |


---

## 3. Solver Details

### Traditional Cpu: Adjoint

- **Module**: `pwm_core.recon.adjoint`
- **Function**: `run_adjoint`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: PnP-ADMM

- **Module**: `pwm_core.recon.pnp_admm`
- **Function**: `pnp_admm_recon`
- **Parameters**: 2M
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
3. Add the solver tier to the modality config in `benchmarks/configs/cars.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
