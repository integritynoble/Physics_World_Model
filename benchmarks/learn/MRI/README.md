# MRI Learning Materials

A self-contained curriculum for understanding MRI physics, inverse problems,
reconstruction algorithms, and the PWM MRI benchmark. Start from the top and
work down -- each file builds on the previous one.

## Prerequisites

Before diving in, you should be comfortable with:

- **Linear algebra**: matrix-vector products, eigenvalues, conjugate transpose
- **Signal processing**: Fourier transform (1D and 2D), convolution, sampling
- **Python / NumPy**: array manipulation, complex numbers, basic plotting
- **Calculus**: gradients, optimisation (gradient descent, proximal operators)

Optional but helpful:

- Familiarity with inverse problems or regularisation theory
- Basic PyTorch (for the deep learning solvers in files 03 and 05)

## Reading Order

| # | File | Topic | Est. Time |
|---|------|-------|-----------|
| 1 | [01_mri_physics_fundamentals.md](01_mri_physics_fundamentals.md) | NMR, Larmor equation, RF excitation, relaxation, gradients, k-space, multi-coil arrays, signal equation | 45 min |
| 2 | [02_mri_as_inverse_problem.md](02_mri_as_inverse_problem.md) | Forward model, PWM 4-knob mismatch model, noise, undersampling, ill-posedness, regularisation | 30 min |
| 3 | [03_reconstruction_algorithms.md](03_reconstruction_algorithms.md) | Zero-filled RSS, SENSE, CS-MRI, PnP-HQS, VarNet, MoDL -- math, code references, comparison | 50 min |
| 4 | [04_pwm_mri_benchmark.md](04_pwm_mri_benchmark.md) | 3-tier structure, data sources, HDF5 format, mismatch ranges, scoring, submission | 25 min |
| 5 | [05_hands_on_tutorial.md](05_hands_on_tutorial.md) | Loading data, running solvers, computing metrics, visualising results | 40 min |

**Total estimated reading time: ~3 hours**

## Suggested Approach

1. Read files 01 and 02 to build intuition for the physics and the
   inverse problem.
2. Read file 03 with a terminal open so you can inspect the referenced
   source files alongside the math.
3. Read file 04 to understand how the benchmark is structured and scored.
4. Work through file 05 as the capstone -- it ties everything together with
   runnable code snippets that load real data, call the solvers, and
   compute metrics.

## Key Source Files

| Module | Path |
|--------|------|
| Classical solvers | `packages/pwm_core/pwm_core/recon/mri_solvers.py` |
| PnP framework | `packages/pwm_core/pwm_core/recon/pnp.py` |
| VarNet | `packages/pwm_core/pwm_core/recon/varnet.py` |
| MoDL | `packages/pwm_core/pwm_core/recon/modl.py` |
| Metrics | `benchmarks/framework/metrics.py` |
| Dataset builder | `datasets/benchmark/mri/build_dataset.py` |
