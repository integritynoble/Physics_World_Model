# Multispectral Satellite Imaging (`multispectral_sat`)

Category: Remote Sensing

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | RDA [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | SAR-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `ms_dl` | MS-Pansharpening-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.multispectral_sat import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.multispectral_sat import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CDFAN | 2024 | 42.8 | 37.9 | partial |
| PanNet | 2017 | 36.1 | 37.9 | done |
| GPPNN | 2021 | 33.8 | 37.9 | done |
| BDSD (Band-Dependent Spatial Detail) | 2008 | 30.0 | 37.9 | done |
| EXP baseline (bicubic LRMS) | 2022 | 27.4 | 37.9 | done |
| Nearest-neighbor (4x) | 2000 | 22.0 | 37.9 | done |
| RDA [proxy] (PWM) | — | 13.9 | 37.9 | done |
| SAR-DL [proxy] (PWM) | — | 13.9 | 37.9 | done |
| MS-Pansharpening-DL [proxy] (PWM) | — | 13.9 | 37.9 | done |
| bicubic_upsample (test) | — | 11.3 | 37.9 | done |
