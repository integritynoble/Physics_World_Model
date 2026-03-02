# 03 — Reconstruction Algorithms: Ptychographic Imaging

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Ptychographic Imaging is **`epie`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | ePIE | `pwm_core.recon.ptychography_solver` | `run_epie` | No |  |
| best_quality | PtychoNN | `pwm_core.recon.ptychonn` | `ptychonn_reconstruct` | No | Cherukara et al. 2020 |
| famous_dl | PtychoNN | `pwm_core.recon.ptychonn` | `ptychonn_reconstruct` | No |  |
| small_gpu | PtychoNN 2.0 | `pwm_core.recon.ptychonn` | `ptychonn_reconstruct` | No |  |


---

## 3. Solver Details

### Traditional Cpu: ePIE

- **Module**: `pwm_core.recon.ptychography_solver`
- **Function**: `run_epie`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: PtychoNN

- **Module**: `pwm_core.recon.ptychonn`
- **Function**: `ptychonn_reconstruct`
- **Parameters**: 4.7M
- **GPU required**: No
- **Reference**: Cherukara et al. 2020

### Famous Dl: PtychoNN

- **Module**: `pwm_core.recon.ptychonn`
- **Function**: `ptychonn_reconstruct`
- **Parameters**: 4.7M
- **GPU required**: No

### Small Gpu: PtychoNN 2.0

- **Module**: `pwm_core.recon.ptychonn`
- **Function**: `ptychonn_reconstruct`
- **Parameters**: 0.7M
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
3. Add the solver tier to the modality config in `benchmarks/configs/ptychography.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
