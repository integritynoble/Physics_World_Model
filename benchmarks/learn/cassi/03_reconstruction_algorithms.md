# 03 — Reconstruction Algorithms: Coded Aperture Snapshot Spectral Imaging (CASSI)

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Coded Aperture Snapshot Spectral Imaging (CASSI) is **`mst`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | GAP-TV | `pwm_core.recon.gap_tv` | `run_gap_tv` | No |  |
| best_quality | HDNet | `pwm_core.recon.hdnet` | `hdnet_recon_cassi` | No | Hu et al. CVPR 2022 |
| famous_dl | MST-L | `pwm_core.recon.mst` | `mst_recon_cassi` | No | Cai et al. CVPR 2022 |
| small_gpu | MST++ | `pwm_core.recon.mst` | `mst_recon_cassi` | No |  |
| pnp_baseline | GAP+HSI-SDeCNN | `pwm_core.recon.gap_tv` | `run_gap_denoise` | No |  |


---

## 3. Solver Details

### Traditional Cpu: GAP-TV

- **Module**: `pwm_core.recon.gap_tv`
- **Function**: `run_gap_tv`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: HDNet

- **Module**: `pwm_core.recon.hdnet`
- **Function**: `hdnet_recon_cassi`
- **Parameters**: 2.37M
- **GPU required**: No
- **Reference**: Hu et al. CVPR 2022

### Famous Dl: MST-L

- **Module**: `pwm_core.recon.mst`
- **Function**: `mst_recon_cassi`
- **Parameters**: 0
- **GPU required**: No
- **Reference**: Cai et al. CVPR 2022

### Small Gpu: MST++

- **Module**: `pwm_core.recon.mst`
- **Function**: `mst_recon_cassi`
- **Parameters**: 1.33M
- **GPU required**: No

### Pnp Baseline: GAP+HSI-SDeCNN

- **Module**: `pwm_core.recon.gap_tv`
- **Function**: `run_gap_denoise`
- **Parameters**: 0.56M
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
3. Add the solver tier to the modality config in `benchmarks/configs/cassi.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
