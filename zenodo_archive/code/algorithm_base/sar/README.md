# Synthetic Aperture Radar (SAR) (`sar`)

Category: Remote Sensing

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP (SAR backprojection) | `pwm_core.recon.ct_solvers.run_fbp` | No |  |
| `best_quality` | SAR-DL (PolSF) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | SAR-CNN [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.sar import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.sar import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Omega-K Algorithm | 1992 | 27.0 | 38.8 | done |
| Range-Doppler Algorithm | 1978 | 25.0 | 38.8 | done |
| SAR-DL (PolSF) [proxy] (PWM) | — | 23.0 | 38.8 | done |
| SAR-CNN [proxy] (PWM) | — | 23.0 | 38.8 | done |
| Matched Filter (192 pts) | 2024 | 19.1 | 38.8 | done |
| FBP (SAR backprojection) (PWM) | — | 18.5 | 38.8 | done |
| precomputed_baseline (test) | — | 18.5 | 38.8 | done |
| Matched Filter (24 pts, 2dB SNR) | 2024 | 8.8 | 38.8 | done |
