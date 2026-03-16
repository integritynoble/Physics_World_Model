# MINFLUX Nanoscopy (`minflux`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `minflux_dl` | MINFLUX-Net | `pwm_core.recon.minflux_solvers.minflux_dl_recon` | Yes | Gwosch, K.C. et al. (2020) MINFLUX nanoscopy 3D, Nature Methods 17:217 |

## Usage

```python
# Import and run
from algorithm_base.minflux import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.minflux import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy (PWM) | — | 29.5 | 34.0 | done |
| CARE (PWM) | — | 29.5 | 34.0 | done |
| MINFLUX-Net (PWM) | — | 29.5 | 34.0 | done |
| precomputed_baseline (test) | — | 29.5 | 34.0 | done |
| MLE localization | 2006 | 18.0 | 34.0 | done |
| Gaussian fitting | 2002 | 15.0 | 34.0 | done |
