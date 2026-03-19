# 03 — Reconstruction Algorithms: Photoacoustic Imaging

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Photoacoustic Imaging is **`back_projection`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | Back Projection | `pwm_core.recon.photoacoustic_solver` | `run_photoacoustic` | No |  |
| best_quality | Time Reversal | `pwm_core.recon.photoacoustic_solver` | `time_reversal_recon` | No | Treeby et al. 2010, J. Biomed. Optics |
| famous_dl | Deep-PAT | `pwm_core.recon.photoacoustic_solver` | `deep_pat_recon` | No | Antholzer et al. 2019, Inverse Problems |
| small_gpu | Deep-PAT | `pwm_core.recon.photoacoustic_solver` | `deep_pat_recon` | No |  |


---

## 3. Solver Details

### Traditional Cpu: Back Projection

- **Module**: `pwm_core.recon.photoacoustic_solver`
- **Function**: `run_photoacoustic`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: Time Reversal

- **Module**: `pwm_core.recon.photoacoustic_solver`
- **Function**: `time_reversal_recon`
- **Parameters**: 0
- **GPU required**: No
- **Reference**: Treeby et al. 2010, J. Biomed. Optics

### Famous Dl: Deep-PAT

- **Module**: `pwm_core.recon.photoacoustic_solver`
- **Function**: `deep_pat_recon`
- **Parameters**: 8M
- **GPU required**: No
- **Reference**: Antholzer et al. 2019, Inverse Problems

### Small Gpu: Deep-PAT

- **Module**: `pwm_core.recon.photoacoustic_solver`
- **Function**: `deep_pat_recon`
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
3. Add the solver tier to the modality config in `benchmarks/configs/photoacoustic.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
