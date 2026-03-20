# 03 — Reconstruction Algorithms: Digital Holographic Microscopy

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Digital Holographic Microscopy is **`angular_spectrum`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | Angular Spectrum | `pwm_core.recon.holography_solver` | `run_holography_reconstruction` | No |  |
| best_quality | PhaseNet | `pwm_core.recon.phasenet` | `phasenet_reconstruct` | No | Rivenson et al. 2018, Light: S&A |
| famous_dl | PhaseNet | `pwm_core.recon.phasenet` | `phasenet_reconstruct` | No |  |
| small_gpu | PhaseNet | `pwm_core.recon.phasenet` | `phasenet_reconstruct` | No |  |


---

## 3. Solver Details

### Traditional Cpu: Angular Spectrum

- **Module**: `pwm_core.recon.holography_solver`
- **Function**: `run_holography_reconstruction`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: PhaseNet

- **Module**: `pwm_core.recon.phasenet`
- **Function**: `phasenet_reconstruct`
- **Parameters**: 2M
- **GPU required**: No
- **Reference**: Rivenson et al. 2018, Light: S&A

### Famous Dl: PhaseNet

- **Module**: `pwm_core.recon.phasenet`
- **Function**: `phasenet_reconstruct`
- **Parameters**: 2M
- **GPU required**: No

### Small Gpu: PhaseNet

- **Module**: `pwm_core.recon.phasenet`
- **Function**: `phasenet_reconstruct`
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
3. Add the solver tier to the modality config in `benchmarks/configs/holography.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
