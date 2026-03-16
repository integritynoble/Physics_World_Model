# Light-Sheet Fluorescence Microscopy (LSFM) (`lightsheet`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Fourier Notch Filter | `pwm_core.recon.lightsheet_solver.run_lightsheet` | No |  |
| `best_quality` | VSNR | `pwm_core.recon.lightsheet_solver.run_lightsheet` | No |  |
| `famous_dl` | DeStripe | `pwm_core.recon.destripe_net.run_destripe` | No | Liang et al. 2022 |
| `small_gpu` | DeStripe | `pwm_core.recon.destripe_net.run_destripe` | No |  |

## Usage

```python
# Import and run
from algorithm_base.lightsheet import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.lightsheet import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CARE | 2018 | 33.0 | 38.5 | done |
| Richardson-Lucy | 1972 | 26.0 | 38.5 | done |
| Fourier Notch Filter (PWM) | — | 23.0 | 38.5 | done |
| VSNR (PWM) | — | 23.0 | 38.5 | done |
| DeStripe (PWM) | — | 23.0 | 38.5 | done |
| precomputed_baseline (test) | — | 23.0 | 38.5 | done |
| rl_20iter (test) | — | 23.0 | 38.5 | done |
| fourier_notch (test) | — | 23.0 | 38.5 | done |
| Gaussian denoising | 2000 | 22.0 | 38.5 | done |
