# 03 — Reconstruction Algorithms: Panorama Multi-Focus Fusion

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Panorama Multi-Focus Fusion is **`laplacian_pyramid_fusion`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | Laplacian Pyramid Fusion | `pwm_core.recon.panorama_solver` | `run_panorama_fusion` | No |  |
| best_quality | Guided Filter Fusion | `pwm_core.recon.panorama_solver` | `multifocus_fusion_guided` | No |  |
| famous_dl | IFCNN | `pwm_core.recon.ifcnn` | `ifcnn_fuse` | No | Zhang et al. 2020 |
| small_gpu | IFCNN | `pwm_core.recon.ifcnn` | `ifcnn_fuse` | No |  |


---

## 3. Solver Details

### Traditional Cpu: Laplacian Pyramid Fusion

- **Module**: `pwm_core.recon.panorama_solver`
- **Function**: `run_panorama_fusion`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: Guided Filter Fusion

- **Module**: `pwm_core.recon.panorama_solver`
- **Function**: `multifocus_fusion_guided`
- **Parameters**: 0
- **GPU required**: No

### Famous Dl: IFCNN

- **Module**: `pwm_core.recon.ifcnn`
- **Function**: `ifcnn_fuse`
- **Parameters**: 0.3M
- **GPU required**: No
- **Reference**: Zhang et al. 2020

### Small Gpu: IFCNN

- **Module**: `pwm_core.recon.ifcnn`
- **Function**: `ifcnn_fuse`
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
3. Add the solver tier to the modality config in `benchmarks/configs/panorama.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
