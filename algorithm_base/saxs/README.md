# Small-Angle X-ray Scattering (SAXS) (`saxs`)

Category: Scientific Instrumentation

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `saxs_dl` | SAXS-VAE [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.saxs import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.saxs import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| McSAS | 2013 | 25.0 | 25.7 | done |
| Guinier analysis | 1939 | 20.0 | 25.7 | done |
| Adjoint [proxy] (PWM) | — | 10.1 | 25.7 | done |
| PnP-ADMM [proxy] (PWM) | — | 10.1 | 25.7 | done |
| SAXS-VAE [proxy] (PWM) | — | 10.1 | 25.7 | done |
| precomputed_baseline (test) | — | 9.0 | 25.7 | done |
