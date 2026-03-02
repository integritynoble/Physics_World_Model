# 03 — Reconstruction Algorithms: Lensless (Diffuser Camera) Imaging

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Lensless (Diffuser Camera) Imaging is **`admm_tv`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | ADMM-TV | `pwm_core.recon.lensless_solver` | `run_lensless` | No | Antipa et al. 2018 |
| best_quality | FlatNet | `pwm_core.recon.flatnet` | `flatnet_reconstruct` | No | Khan et al. TPAMI 2020 |
| famous_dl | FlatNet | `pwm_core.recon.flatnet` | `flatnet_reconstruct` | No |  |
| small_gpu | FlatNet-Lite | `pwm_core.recon.flatnet` | `flatnet_reconstruct` | No |  |


---

## 3. Solver Details

### Traditional Cpu: ADMM-TV

- **Module**: `pwm_core.recon.lensless_solver`
- **Function**: `run_lensless`
- **Parameters**: 0
- **GPU required**: No
- **Reference**: Antipa et al. 2018

### Best Quality: FlatNet

- **Module**: `pwm_core.recon.flatnet`
- **Function**: `flatnet_reconstruct`
- **Parameters**: 59M
- **GPU required**: No
- **Reference**: Khan et al. TPAMI 2020

### Famous Dl: FlatNet

- **Module**: `pwm_core.recon.flatnet`
- **Function**: `flatnet_reconstruct`
- **Parameters**: 59M
- **GPU required**: No

### Small Gpu: FlatNet-Lite

- **Module**: `pwm_core.recon.flatnet`
- **Function**: `flatnet_reconstruct`
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
3. Add the solver tier to the modality config in `benchmarks/configs/lensless.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
