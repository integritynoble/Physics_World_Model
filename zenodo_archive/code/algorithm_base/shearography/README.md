# Shearography (`shearography`)

Category: Industrial Inspection

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `shear_dl` | ShearNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.shearography import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.shearography import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Phase-shifting shearography | 2000 | 28.0 | 36.5 | done |
| FPD-CNN | 2020 | 27.9 | 36.5 | done |
| Fourier transform method | 1982 | 25.0 | 36.5 | done |
| DBDNet | 2021 | 20.6 | 36.5 | done |
| Adjoint [proxy] (PWM) | — | 19.1 | 36.5 | done |
| PnP-ADMM [proxy] (PWM) | — | 19.1 | 36.5 | done |
| ShearNet [proxy] (PWM) | — | 19.1 | 36.5 | done |
| OCPDE (Oriented Coupled PDE) | 2020 | 14.1 | 36.5 | done |
| precomputed_baseline (test) | — | 13.2 | 36.5 | done |
| WFLPF (Windowed Fourier LP Filter) | 2020 | 12.8 | 36.5 | done |
