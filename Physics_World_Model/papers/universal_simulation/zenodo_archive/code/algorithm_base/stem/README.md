# Scanning Transmission Electron Microscopy (STEM) (`stem`)

Category: Electron Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy (STEM) | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | STEM-DL (AtomSegNet) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | STEM-UNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.stem import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.stem import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DAE (Denoising AE) | 2023 | 42.9 | 25.7 | gap |
| STEM-DL (AtomSegNet) [proxy] (PWM) | — | 36.2 | 25.7 | gap |
| STEM-UNet [proxy] (PWM) | — | 36.2 | 25.7 | gap |
| Richardson-Lucy (STEM) (PWM) | — | 34.5 | 25.7 | partial |
| precomputed_baseline (test) | — | 34.5 | 25.7 | partial |
| SwinIR | 2021 | 33.0 | 25.7 | partial |
| BM3D | 2007 | 30.0 | 25.7 | partial |
