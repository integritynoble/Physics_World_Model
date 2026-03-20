# Electron Tomography (`electron_tomography`)

Category: Electron Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP (SIRT baseline) | `pwm_core.recon.ct_solvers.run_fbp` | No |  |
| `best_quality` | IMOD-SIRT-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | SIRT-3D [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.electron_tomography import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.electron_tomography import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Joint DL model (IRDM) | 2019 | 27.5 | 24.1 | partial |
| IMOD-SIRT-DL [proxy] (PWM) | — | 26.1 | 24.1 | done |
| SIRT-3D [proxy] (PWM) | — | 26.1 | 24.1 | done |
| FBP (SIRT baseline) (PWM) | — | 25.1 | 24.1 | done |
| precomputed_baseline (test) | — | 25.1 | 24.1 | done |
| SART (missing wedge) | 1972 | 18.6 | 24.1 | done |
| WBP (missing wedge) | 1970 | 13.1 | 24.1 | done |
