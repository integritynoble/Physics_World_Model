# Diffuse Optical Tomography (DOT) (`dot`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Born Approximation | `pwm_core.recon.dot_solver.run_dot` | No |  |
| `best_quality` | L-BFGS-TV [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `dot_dl` | DOT-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.dot import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.dot import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| BPNN | 2018 | 27.8 | 27.1 | done |
| Tikhonov regularization | 2018 | 24.3 | 27.1 | done |
| Tikhonov (basic, noisy) | 2000 | 22.0 | 27.1 | done |
| Born approximation | 1999 | 20.0 | 27.1 | done |
| Rytov + Laplacian | 2000 | 18.0 | 27.1 | done |
| L-BFGS-TV [proxy] (PWM) | — | 8.0 | 27.1 | done |
| DOT-Net [proxy] (PWM) | — | 8.0 | 27.1 | done |
| born_backprojection (test) | — | 7.0 | 27.1 | done |
| tikhonov (test) | — | 7.0 | 27.1 | done |
| precomputed_baseline (test) | — | 7.0 | 27.1 | done |
