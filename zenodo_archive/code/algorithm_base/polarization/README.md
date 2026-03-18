# Polarization Microscopy (`polarization`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | PnP-HQS | `pwm_core.recon.pnp.run_pnp` | No |  |
| `best_quality` | PolarNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | Stokes-NN [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.polarization import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.polarization import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| PolarNet [proxy] (PWM) | — | 47.8 | 30.1 | gap |
| Stokes-NN [proxy] (PWM) | — | 47.8 | 30.1 | gap |
| MDU-Net | 2022 | 38.1 | 30.1 | partial |
| MIRNet | 2022 | 37.9 | 30.1 | partial |
| DnCNN | 2022 | 34.4 | 30.1 | partial |
| PnP-HQS (PWM) | — | 30.9 | 30.1 | done |
| precomputed_baseline (test) | — | 30.9 | 30.1 | done |
| Raw Mueller matrix | 2022 | 29.0 | 30.1 | done |
| Mueller matrix | 2000 | 25.0 | 30.1 | done |
