# XFEL Serial Femtosecond Crystallography (SFX) (`xfel_sfx`)

Category: Ultrafast Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `sfx_dl` | SFX-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.xfel_sfx import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.xfel_sfx import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 25.1 | 22.8 | done |
| PnP-ADMM [proxy] (PWM) | — | 25.1 | 22.8 | done |
| SFX-Net [proxy] (PWM) | — | 25.1 | 22.8 | done |
| cctbx.xfel | 2014 | 25.0 | 22.8 | done |
| precomputed_baseline (test) | — | 24.1 | 22.8 | done |
| CrystFEL | 2012 | 22.0 | 22.8 | done |
