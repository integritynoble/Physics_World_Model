# CT + Fluorescence (FLIT) (`ct_fluorescence`)

Category: Multi-Modal Fusion

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `xfct_dl` | XFCT-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.ct_fluorescence import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ct_fluorescence import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SIRT | 1972 | 25.0 | 34.0 | done |
| FBP + fluorescence | 2000 | 22.0 | 34.0 | done |
| Adjoint [proxy] (PWM) | — | 11.2 | 34.0 | done |
| PnP-ADMM [proxy] (PWM) | — | 11.2 | 34.0 | done |
| XFCT-Net [proxy] (PWM) | — | 11.2 | 34.0 | done |
| precomputed_baseline (test) | — | 10.2 | 34.0 | done |
