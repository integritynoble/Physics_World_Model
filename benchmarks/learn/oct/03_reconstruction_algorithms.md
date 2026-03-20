# 03 — Reconstruction Algorithms: Optical Coherence Tomography (OCT)

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Optical Coherence Tomography (OCT) is **`fft_recon`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | FFT Recon | `pwm_core.recon.oct_solver` | `run_oct` | No |  |
| best_quality | Spectral Estimation | `pwm_core.recon.oct_solver` | `spectral_estimation_recon` | No | Leitgeb et al. 2003, Optics Express |
| famous_dl | OCT Denoising Net | `pwm_core.recon.oct_solver` | `oct_denoising_net_recon` | No | Devalla et al. 2019, Biomed. Optics Express |
| small_gpu | OCT Denoising Net | `pwm_core.recon.oct_solver` | `oct_denoising_net_recon` | No |  |


---

## 3. Solver Details

### Traditional Cpu: FFT Recon

- **Module**: `pwm_core.recon.oct_solver`
- **Function**: `run_oct`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: Spectral Estimation

- **Module**: `pwm_core.recon.oct_solver`
- **Function**: `spectral_estimation_recon`
- **Parameters**: 0
- **GPU required**: No
- **Reference**: Leitgeb et al. 2003, Optics Express

### Famous Dl: OCT Denoising Net

- **Module**: `pwm_core.recon.oct_solver`
- **Function**: `oct_denoising_net_recon`
- **Parameters**: 5M
- **GPU required**: No
- **Reference**: Devalla et al. 2019, Biomed. Optics Express

### Small Gpu: OCT Denoising Net

- **Module**: `pwm_core.recon.oct_solver`
- **Function**: `oct_denoising_net_recon`
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
3. Add the solver tier to the modality config in `benchmarks/configs/oct.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
