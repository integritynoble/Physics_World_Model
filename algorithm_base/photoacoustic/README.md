# Photoacoustic Imaging (`photoacoustic`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Back Projection | `pwm_core.recon.photoacoustic_solver.run_photoacoustic` | No |  |
| `best_quality` | Time Reversal [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | Deep-PAT [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `small_gpu` | Deep-PAT [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.photoacoustic import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.photoacoustic import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Iterative (model-based) | 2000 | 30.2 | 19.0 | gap |
| Residual U-Net (Deep-PAT) | 2021 | 29.9 | 19.0 | gap |
| Pixel-DL | 2020 | 29.6 | 19.0 | gap |
| Post-DL (U-Net) | 2020 | 24.4 | 19.0 | partial |
| Time Reversal (FBP) | 2000 | 22.7 | 19.0 | partial |
| Backprojection (limited view) | 2021 | 21.9 | 19.0 | done |
| Deep-PAT [proxy] (PWM) | — | 21.2 | 19.0 | done |
| Back Projection (PWM) | — | 19.8 | 19.0 | done |
| precomputed_baseline (test) | — | 19.8 | 19.0 | done |
| Time Reversal (16 sensors) | 2020 | 13.9 | 19.0 | done |
| Tikhonov (32 views) | 2023 | 13.9 | 19.0 | done |
