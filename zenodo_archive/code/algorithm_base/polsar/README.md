# Polarimetric SAR (PolSAR) (`polsar`)

Category: Remote Sensing

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | RDA [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | SAR-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `polsar_dl` | PolSAR-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.polsar import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.polsar import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| PAN-DeSpeck | 2023 | 28.4 | 39.6 | done |
| CNN learnable activation | 2021 | 26.4 | 39.6 | done |
| Refined Lee | 2003 | 24.0 | 39.6 | done |
| Cloude-Pottier decomposition | 1997 | 22.3 | 39.6 | done |
| RDA [proxy] (PWM) | — | 22.3 | 39.6 | done |
| SAR-DL [proxy] (PWM) | — | 22.3 | 39.6 | done |
| PolSAR-Net [proxy] (PWM) | — | 22.3 | 39.6 | done |
| Lee filter | 1999 | 22.0 | 39.6 | done |
| precomputed_baseline (test) | — | 19.4 | 39.6 | done |
| Single-look noisy input | 2017 | 14.5 | 39.6 | done |
