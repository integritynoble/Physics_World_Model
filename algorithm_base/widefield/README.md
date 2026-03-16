# Widefield Fluorescence Microscopy (`widefield`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018, Nature Methods |
| `famous_dl` | CARE | `pwm_core.recon.care_unet.run_care` | No |  |
| `small_gpu` | CARE | `pwm_core.recon.care_unet.run_care` | No |  |

## Usage

```python
# Import and run
from algorithm_base.widefield import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.widefield import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Restormer | 2022 | 35.5 | 47.8 | done |
| Noise2Void | 2019 | 31.0 | 47.8 | done |
| Wiener deconvolution | 1949 | 26.0 | 47.8 | done |
| precomputed_baseline (test) | — | 25.0 | 47.8 | done |
| m-rBCR | 2023 | 24.9 | 47.8 | done |
| CARE | 2018 | 22.1 | 47.8 | done |
| Richardson-Lucy (20 iter) | 1972 | 13.4 | 47.8 | done |
