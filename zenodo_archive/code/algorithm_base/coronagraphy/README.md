# Stellar Coronagraphy (`coronagraphy`)

Category: Astronomy & Space Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `speckle_null_dl` | DL-SpeckleNull [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.coronagraphy import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.coronagraphy import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 28.8 | 37.9 | done |
| PnP-ADMM [proxy] (PWM) | — | 28.8 | 37.9 | done |
| DL-SpeckleNull [proxy] (PWM) | — | 28.8 | 37.9 | done |
| precomputed_baseline (test) | — | 27.7 | 37.9 | done |
| PCA/KLIP | 2012 | 22.0 | 37.9 | done |
| LOCI | 2007 | 20.0 | 37.9 | done |
| Classical ADI | 2006 | 18.0 | 37.9 | done |
