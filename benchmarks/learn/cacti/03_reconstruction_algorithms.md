# 03 — Reconstruction Algorithms: Coded Aperture Compressive Temporal Imaging (CACTI)

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Coded Aperture Compressive Temporal Imaging (CACTI) is **`gap_tv`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | GAP-TV | `pwm_core.recon.gap_tv` | `run_gap_tv` | No |  |
| best_quality | EfficientSCI | `pwm_core.recon.efficientsci` | `efficientsci_recon` | No | Wang et al. CVPR 2023 |
| famous_dl | ELP-Unfolding | `pwm_core.recon.elp_unfolding` | `elp_recon` | No | Yang et al. ECCV 2022 |
| small_gpu | EfficientSCI-T | `pwm_core.recon.efficientsci` | `efficientsci_recon` | No |  |


---

## 3. Solver Details

### Traditional Cpu: GAP-TV

- **Module**: `pwm_core.recon.gap_tv`
- **Function**: `run_gap_tv`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: EfficientSCI

- **Module**: `pwm_core.recon.efficientsci`
- **Function**: `efficientsci_recon`
- **Parameters**: 12.05M
- **GPU required**: No
- **Reference**: Wang et al. CVPR 2023

### Famous Dl: ELP-Unfolding

- **Module**: `pwm_core.recon.elp_unfolding`
- **Function**: `elp_recon`
- **Parameters**: 10M
- **GPU required**: No
- **Reference**: Yang et al. ECCV 2022

### Small Gpu: EfficientSCI-T

- **Module**: `pwm_core.recon.efficientsci`
- **Function**: `efficientsci_recon`
- **Parameters**: 3.78M
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
3. Add the solver tier to the modality config in `benchmarks/configs/cacti.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
