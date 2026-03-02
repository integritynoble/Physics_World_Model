# 03 — Reconstruction Algorithms: Single-Pixel Camera (SPC)

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Single-Pixel Camera (SPC) is **`pnp_fista`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | TVAL3 | `pwm_core.recon.cs_solvers` | `run_tval3` | No |  |
| best_quality | HATNet | `pwm_core.recon.hatnet` | `hatnet_reconstruct` | No | Qu et al. CVPR 2024 |
| famous_dl | ISTA-Net+ | `pwm_core.recon.ista_net` | `ista_net_reconstruct` | No | Zhang & Ghanem, CVPR 2018 |
| small_gpu | ISTA-Net+ | `pwm_core.recon.ista_net` | `ista_net_reconstruct` | No |  |


---

## 3. Solver Details

### Traditional Cpu: TVAL3

- **Module**: `pwm_core.recon.cs_solvers`
- **Function**: `run_tval3`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: HATNet

- **Module**: `pwm_core.recon.hatnet`
- **Function**: `hatnet_reconstruct`
- **Parameters**: 8M
- **GPU required**: No
- **Reference**: Qu et al. CVPR 2024

### Famous Dl: ISTA-Net+

- **Module**: `pwm_core.recon.ista_net`
- **Function**: `ista_net_reconstruct`
- **Parameters**: 3M
- **GPU required**: No
- **Reference**: Zhang & Ghanem, CVPR 2018

### Small Gpu: ISTA-Net+

- **Module**: `pwm_core.recon.ista_net`
- **Function**: `ista_net_reconstruct`
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
3. Add the solver tier to the modality config in `benchmarks/configs/spc.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
