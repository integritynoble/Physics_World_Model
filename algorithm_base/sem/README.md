# Scanning Electron Microscopy (SEM) (`sem`)

Category: Electron Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy (SEM) | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | SEM-DL (SegNet) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | SEM-UNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.sem import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.sem import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SwinIR | 2021 | 34.0 | 39.0 | done |
| BM3D | 2007 | 30.0 | 39.0 | done |
| SEM-DL (SegNet) [proxy] (PWM) | — | 28.8 | 39.0 | done |
| SEM-UNet [proxy] (PWM) | — | 28.8 | 39.0 | done |
| Noise2Void | 2019 | 28.0 | 39.0 | done |
| NLM | 2005 | 25.0 | 39.0 | done |
| Richardson-Lucy (SEM) (PWM) | — | 23.2 | 39.0 | done |
| precomputed_baseline (test) | — | 23.2 | 39.0 | done |
| Gaussian filter | 2000 | 22.0 | 39.0 | done |
