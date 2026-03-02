# 03 — Reconstruction Algorithms: Low-Dose Widefield Microscopy

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Low-Dose Widefield Microscopy is **`pnp_hqs`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | BM3D + RL | `pwm_core.recon.richardson_lucy` | `run_richardson_lucy` | No |  |
| best_quality | CARE | `pwm_core.recon.care_unet` | `care_restore_2d` | Yes |  |
| famous_dl | Noise2Void | `pwm_core.recon.noise2void` | `noise2void_denoise` | No | Krull et al. CVPR 2019 |
| small_gpu | Noise2Void | `pwm_core.recon.noise2void` | `noise2void_denoise` | No |  |


---

## 3. Solver Details

### Traditional Cpu: BM3D + RL

- **Module**: `pwm_core.recon.richardson_lucy`
- **Function**: `run_richardson_lucy`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: CARE

- **Module**: `pwm_core.recon.care_unet`
- **Function**: `care_restore_2d`
- **Parameters**: 2M
- **GPU required**: Yes

### Famous Dl: Noise2Void

- **Module**: `pwm_core.recon.noise2void`
- **Function**: `noise2void_denoise`
- **Parameters**: 1M
- **GPU required**: No
- **Reference**: Krull et al. CVPR 2019

### Small Gpu: Noise2Void

- **Module**: `pwm_core.recon.noise2void`
- **Function**: `noise2void_denoise`
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
3. Add the solver tier to the modality config in `benchmarks/configs/widefield_lowdose.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
