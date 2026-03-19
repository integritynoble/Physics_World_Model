# Wide-Angle X-ray Scattering (WAXS) (`waxs`)

Category: Scientific Instrumentation

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `waxs_dl` | WAXS-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.waxs import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.waxs import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 24.5 | 24.0 | done |
| PnP-ADMM [proxy] (PWM) | — | 24.5 | 24.0 | done |
| WAXS-Net [proxy] (PWM) | — | 24.5 | 24.0 | done |
| Rietveld refinement | 1969 | 24.0 | 24.0 | done |
| precomputed_baseline (test) | — | 23.4 | 24.0 | done |
| Background subtraction | 2000 | 20.0 | 24.0 | done |
