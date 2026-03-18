# Atomic Force Microscopy (AFM) (`afm`)

Category: Scanning Probe Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `afm_dl` | AFM-UNet | `pwm_core.recon.afm_solvers.afm_unet_recon` | Yes | Cherukara, M.J. et al. (2020) AI-enabled high-res, real-time imaging, npj Comput. Mater. 6:203 |

## Usage

```python
# Import and run
from algorithm_base.afm import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.afm import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Deep-AFM | 2020 | 32.0 | 44.9 | done |
| Richardson-Lucy (PWM) | — | 31.3 | 44.9 | done |
| CARE (PWM) | — | 31.3 | 44.9 | done |
| AFM-UNet (PWM) | — | 31.3 | 44.9 | done |
| precomputed_baseline (test) | — | 31.3 | 44.9 | done |
| Flatten + line correction | 2000 | 25.0 | 44.9 | done |
