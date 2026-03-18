# Image Scanning Microscopy (ISM) (`ism`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `ism_dl` | ISM-Reassignment-Net | `pwm_core.recon.ism_solvers.ism_dl_recon` | Yes | Castello, M. et al. (2019) Image scanning microscopy ISM, Nature Methods 16:175 |

## Usage

```python
# Import and run
from algorithm_base.ism import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ism import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy (PWM) | — | 34.0 | 33.0 | done |
| CARE (PWM) | — | 34.0 | 33.0 | done |
| ISM-Reassignment-Net (PWM) | — | 34.0 | 33.0 | done |
| precomputed_baseline (test) | — | 34.0 | 33.0 | done |
| Airyscan processing | 2017 | 30.0 | 33.0 | done |
| Pixel reassignment | 2010 | 28.0 | 33.0 | done |
