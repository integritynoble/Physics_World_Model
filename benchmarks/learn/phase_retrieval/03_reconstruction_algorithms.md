# 03 — Reconstruction Algorithms: Coherent Diffractive Imaging / Phase Retrieval

## 1. Overview

The PWM benchmark evaluates reconstruction algorithms at multiple quality
tiers, from classical CPU-only methods to state-of-the-art deep learning.
The default solver for Coherent Diffractive Imaging / Phase Retrieval is **`hio`**.

---

## 2. Solver Comparison Table

| Tier | Name | Module | Function | GPU | Reference |
|------|------|--------|----------|-----|-----------|
| traditional_cpu | HIO | `pwm_core.recon.phase_retrieval_solver` | `run_phase_retrieval` | No |  |
| best_quality | RAAR | `pwm_core.recon.phase_retrieval_solver` | `raar_recon` | No | Luke 2005, Inverse Problems |
| famous_dl | prDeep | `pwm_core.recon.phase_retrieval_solver` | `prdeep_recon` | No | Metzler et al. NeurIPS 2018 |
| small_gpu | prDeep | `pwm_core.recon.phase_retrieval_solver` | `prdeep_recon` | No |  |


---

## 3. Solver Details

### Traditional Cpu: HIO

- **Module**: `pwm_core.recon.phase_retrieval_solver`
- **Function**: `run_phase_retrieval`
- **Parameters**: 0
- **GPU required**: No

### Best Quality: RAAR

- **Module**: `pwm_core.recon.phase_retrieval_solver`
- **Function**: `raar_recon`
- **Parameters**: 0
- **GPU required**: No
- **Reference**: Luke 2005, Inverse Problems

### Famous Dl: prDeep

- **Module**: `pwm_core.recon.phase_retrieval_solver`
- **Function**: `prdeep_recon`
- **Parameters**: 6M
- **GPU required**: No
- **Reference**: Metzler et al. NeurIPS 2018

### Small Gpu: prDeep

- **Module**: `pwm_core.recon.phase_retrieval_solver`
- **Function**: `prdeep_recon`
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
3. Add the solver tier to the modality config in `benchmarks/configs/phase_retrieval.yaml`
4. Run the benchmark to compare against existing solvers

---

*Previous: [02 — Forward Model](02_forward_model.md)*
*Next: [04 — PWM Benchmark](04_pwm_benchmark.md)*
