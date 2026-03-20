# Seismic Tomography (`seismic_tomo`)

Category: Broader Experimental Science

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `seismic_dl` | SeisInversion-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.seismic_tomo import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.seismic_tomo import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| TSISTA-Net | 2025 | 37.3 | 30.6 | partial |
| PhaseNet-DAS | 2023 | 30.0 | 30.6 | done |
| FWI | 2009 | 28.0 | 30.6 | done |
| Travel-time tomography | 1976 | 20.0 | 30.6 | done |
| Simple ray tracing | 1976 | 12.0 | 30.6 | done |
| Adjoint [proxy] (PWM) | — | 11.2 | 30.6 | done |
| PnP-ADMM [proxy] (PWM) | — | 11.2 | 30.6 | done |
| SeisInversion-Net [proxy] (PWM) | — | 11.2 | 30.6 | done |
| precomputed_baseline (test) | — | 9.8 | 30.6 | done |
