# Interferometric SAR (InSAR) (`insar`)

Category: Remote Sensing

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | RDA [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | SAR-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `insar_dl` | InSAR-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.insar import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.insar import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| RDA [proxy] (PWM) | — | 32.8 | 17.4 | gap |
| SAR-DL [proxy] (PWM) | — | 32.8 | 17.4 | gap |
| InSAR-Net [proxy] (PWM) | — | 32.8 | 17.4 | gap |
| wrapped_phase_baseline (test) | — | 31.8 | 17.4 | gap |
| SNAPHU | 2001 | 28.0 | 17.4 | gap |
| Goldstein filter | 1998 | 22.0 | 17.4 | partial |
