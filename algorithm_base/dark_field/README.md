# Dark-Field Microscopy (`dark_field`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `df_unet` | DF-UNet | `pwm_core.recon.darkfield_solvers.df_unet_recon` | Yes | Wolfer, T. et al. (2021) DL for dark-field X-ray CT, Sci. Rep. 11:5005 |

## Usage

```python
# Import and run
from algorithm_base.dark_field import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.dark_field import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DAPD | 2024 | 33.0 | 23.9 | partial |
| BM3D | 2007 | 30.0 | 23.9 | partial |
| Richardson-Lucy (PWM) | — | 25.1 | 23.9 | done |
| CARE (PWM) | — | 25.1 | 23.9 | done |
| DF-UNet (PWM) | — | 25.1 | 23.9 | done |
| precomputed_baseline (test) | — | 25.1 | 23.9 | done |
| Median filter | 2000 | 24.0 | 23.9 | done |
