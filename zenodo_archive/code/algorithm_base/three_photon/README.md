# Three-Photon Microscopy (`three_photon`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `3p_dl` | 3P-Net (CARE) | `pwm_core.recon.three_photon_solvers.three_photon_care_recon` | Yes | Weigert, M. et al. (2018) CARE for 3P deep tissue imaging, Nature Methods 15:1090 |

## Usage

```python
# Import and run
from algorithm_base.three_photon import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.three_photon import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DeepCAD-RT | 2023 | 34.0 | 30.4 | partial |
| Richardson-Lucy | 1972 | 26.0 | 30.4 | done |
| CARE (PWM) | — | 22.3 | 30.4 | done |
| 3P-Net (CARE) (PWM) | — | 22.3 | 30.4 | done |
| precomputed_baseline (test) | — | 22.3 | 30.4 | done |
| Gaussian denoising | 2000 | 20.0 | 30.4 | done |
