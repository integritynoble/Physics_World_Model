# 03 — Reconstruction Algorithms: Structured Illumination Microscopy (SIM)

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Structured Illumination Microscopy (SIM) is **`wiener_sim`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | Wiener-SIM | `pwm_core.recon.sim_solver` | `run_sim_reconstruction` | No |  |
| best_quality | HiFi-SIM | `pwm_core.recon.sim_solver` | `hifi_sim_2d` | No | Wen et al. 2021, Light: S&A |
| famous_dl | DL-SIM | `pwm_core.recon.dl_sim` | `dl_sim_reconstruct` | No | Jin et al. 2020, Nature Comm. |
| small_gpu | DL-SIM | `pwm_core.recon.dl_sim` | `dl_sim_reconstruct` | No |  |


---

## 3. Solver Details

### Traditional Cpu: Wiener-SIM

- **Module**: `pwm_core.recon.sim_solver`
- **Function**: `run_sim_reconstruction`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: HiFi-SIM

- **Module**: `pwm_core.recon.sim_solver`
- **Function**: `hifi_sim_2d`
- **Parameters**: 0
- **GPU required**: No
- **Reference**: Wen et al. 2021, Light: S&A

### Famous Dl: DL-SIM

- **Module**: `pwm_core.recon.dl_sim`
- **Function**: `dl_sim_reconstruct`
- **Parameters**: 3M
- **GPU required**: No
- **Reference**: Jin et al. 2020, Nature Comm.

### Small Gpu: DL-SIM

- **Module**: `pwm_core.recon.dl_sim`
- **Function**: `dl_sim_reconstruct`
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
3. Add the solver tier to the modality config in `benchmarks/configs/sim.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
