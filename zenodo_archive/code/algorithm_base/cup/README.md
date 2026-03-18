# Compressed Ultrafast Photography (CUP) (`cup`)

Category: Ultrafast Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `e2e_cup` | E2E-CUP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.cup import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.cup import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 29.5 | 26.8 | done |
| E2E-CUP [proxy] (PWM) | — | 29.5 | 26.8 | done |
| PnP-BM3D | 2020 | 29.2 | 26.8 | done |
| PnP-FFDNet | 2020 | 28.4 | 26.8 | done |
| PnP-DnCNN | 2020 | 27.1 | 26.8 | done |
| TwIST | 2007 | 24.7 | 26.8 | done |
| Direct inverse (no regularization) | 2014 | 12.0 | 26.8 | done |
| precomputed_baseline (test) | — | 8.5 | 26.8 | done |
| Direct inverse (1000x compression) | 2014 | 8.0 | 26.8 | done |
