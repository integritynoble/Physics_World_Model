# Hyperspectral Remote Sensing (`hyperspectral_remote`)

Category: Remote Sensing

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | RDA [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | SAR-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `hyper_dl` | SST-USRNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.hyperspectral_remote import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.hyperspectral_remote import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| RDA [proxy] (PWM) | — | 49.7 | 40.1 | partial |
| SAR-DL [proxy] (PWM) | — | 49.7 | 40.1 | partial |
| SST-USRNet [proxy] (PWM) | — | 49.7 | 40.1 | partial |
| precomputed_baseline (test) | — | 35.0 | 40.1 | done |
| MST++ | 2022 | 34.3 | 40.1 | done |
| HDNet | 2022 | 32.1 | 40.1 | done |
| AWAN | 2020 | 31.2 | 40.1 | done |
| HSCNN+ | 2018 | 26.4 | 40.1 | done |
