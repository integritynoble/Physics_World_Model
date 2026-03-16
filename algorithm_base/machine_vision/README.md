# Machine Vision / AOI (`machine_vision`)

Category: Industrial Inspection

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `mv_dl` | PatchCore [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.machine_vision import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.machine_vision import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 36.2 | 34.3 | done |
| PnP-ADMM [proxy] (PWM) | — | 36.2 | 34.3 | done |
| UniAD | 2023 | 32.0 | 34.3 | done |
| PatchCore | 2022 | 30.0 | 34.3 | done |
| precomputed_baseline (test) | — | 28.3 | 34.3 | done |
| Template matching | 2000 | 25.0 | 34.3 | done |
