# Weather / Doppler Radar (`weather_radar`)

Category: Remote Sensing

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | RDA [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | SAR-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `weather_dl` | NowcastNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.weather_radar import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.weather_radar import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Axial-UNet | 2025 | 47.7 | 16.8 | gap |
| U-Net | 2020 | 35.0 | 16.8 | gap |
| RDA [proxy] (PWM) | — | 30.2 | 16.8 | gap |
| SAR-DL [proxy] (PWM) | — | 30.2 | 16.8 | gap |
| NowcastNet [proxy] (PWM) | — | 30.2 | 16.8 | gap |
| precomputed_baseline (test) | — | 26.9 | 16.8 | gap |
| CLEAN-AP | 2000 | 25.0 | 16.8 | partial |
