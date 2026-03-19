# Time-of-Flight Depth Camera — Learning Materials

A self-contained curriculum for understanding the physics, forward model,
reconstruction algorithms, and PWM benchmark for **Time-of-Flight Depth Camera**.

## Quick Facts

| Property | Value |
|----------|-------|
| Modality ID | `tof_camera` |
| Category | Depth Imaging |
| Carrier | Photon/IR |
| Forward Model | nonlinear_operator |
| Default Solver | `tv_fista` |
| Maturity | M0 |
| Tier | A |

## Reading Order

| # | File | Topic | Est. Time |
|---|------|-------|-----------|
| 1 | [01_physics_fundamentals.md](01_physics_fundamentals.md) | Physics of Photon/IR imaging, key equations, hardware | 30 min |
| 2 | [02_forward_model.md](02_forward_model.md) | Forward model, mismatch parameters, noise model | 20 min |
| 3 | [03_reconstruction_algorithms.md](03_reconstruction_algorithms.md) | Available solvers, comparison, trade-offs | 25 min |
| 4 | [04_pwm_benchmark.md](04_pwm_benchmark.md) | PWM benchmark structure, data format, scoring | 15 min |
| 5 | [05_hands_on_tutorial.md](05_hands_on_tutorial.md) | Code snippets: load data, run solvers, compute metrics | 20 min |

**Total estimated reading time: ~2 hours**

## Key Source Files

| Module | Path |
|--------|------|
| Benchmark config | `benchmarks/configs/tof_camera.yaml` |
| Modality registry | `packages/pwm_core/contrib/modalities.yaml` |
| Solver registry | `packages/pwm_core/contrib/solver_registry.yaml` |
