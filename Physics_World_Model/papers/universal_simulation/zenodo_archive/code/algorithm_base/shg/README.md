# Second Harmonic Generation (SHG) Microscopy (`shg`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `shg_dl` | SHG-CARE | `pwm_core.recon.shg_solvers.shg_care_recon` | Yes | Weigert, M. et al. (2018) CARE for SHG imaging, Nature Methods 15:1090 |

## Usage

```python
# Import and run
from algorithm_base.shg import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.shg import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy | 1972 | 28.0 | 35.9 | done |
| DnCNN | 2023 | 25.4 | 35.9 | done |
| CARE (PWM) | — | 24.1 | 35.9 | done |
| SHG-CARE (PWM) | — | 24.1 | 35.9 | done |
| precomputed_baseline (test) | — | 24.1 | 35.9 | done |
| Gaussian denoising | 2000 | 22.0 | 35.9 | done |
