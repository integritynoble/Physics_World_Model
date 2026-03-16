# STED Microscopy (`sted`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy (STED) | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | STED-Net (CARE) | `pwm_core.recon.sted_solvers.sted_care_recon` | Yes | Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090 |
| `famous_dl` | RCAN-STED | `pwm_core.recon.sted_solvers.rcan_sted_recon` | Yes | Chen, J. et al. (2021) Three-dimensional residual channel attention for STED, Nature Methods 18:678 |

## Usage

```python
# Import and run
from algorithm_base.sted import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.sted import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DDPM denoiser | 2023 | 32.8 | 35.0 | done |
| STED-Net (CARE) (PWM) | — | 29.6 | 35.0 | done |
| RCAN-STED (PWM) | — | 29.6 | 35.0 | done |
| precomputed_baseline (test) | — | 29.6 | 35.0 | done |
| rl_20iter (test) | — | 29.6 | 35.0 | done |
| Richardson-Lucy STED | 2006 | 28.0 | 35.0 | done |
| Gaussian denoising | 2000 | 24.0 | 35.0 | done |
