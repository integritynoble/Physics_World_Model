# Event Horizon Telescope (EHT) Imaging (`eht_imaging`)

Category: Astronomy & Space Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `eht_dl` | EHT-PRIMO [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.eht_imaging import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.eht_imaging import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| PRIMO | 2023 | 28.0 | 43.1 | done |
| eht-imaging RML | 2019 | 25.0 | 43.1 | done |
| SMILI | 2019 | 24.0 | 43.1 | done |
| CLEAN | 1974 | 20.0 | 43.1 | done |
| Adjoint [proxy] (PWM) | — | 13.0 | 43.1 | done |
| PnP-ADMM [proxy] (PWM) | — | 13.0 | 43.1 | done |
| Dirty beam (no deconvolution) | 1974 | 12.0 | 43.1 | done |
| precomputed_baseline (test) | — | 11.4 | 43.1 | done |
