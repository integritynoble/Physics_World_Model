# Ocean Acoustic Tomography (`ocean_acoustic_tomo`)

Category: Broader Experimental Science

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `oat_dl` | OAT-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.ocean_acoustic_tomo import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ocean_acoustic_tomo import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 27.6 | 33.6 | done |
| PnP-ADMM [proxy] (PWM) | — | 27.6 | 33.6 | done |
| OAT-Net [proxy] (PWM) | — | 27.6 | 33.6 | done |
| precomputed_baseline (test) | — | 26.6 | 33.6 | done |
| Matched-field processing | 1990 | 22.0 | 33.6 | done |
| Travel-time inversion | 1979 | 20.0 | 33.6 | done |
