# Differential Interference Contrast (DIC) (`dic`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `dic_dl` | DIC-Net | `pwm_core.recon.dic_solvers.dic_dl_recon` | Yes | Mir, A. et al. (2015) Automated DIC microscopy, J. Microsc. 257(2) |

## Usage

```python
# Import and run
from algorithm_base.dic import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.dic import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DL phase recovery | 2020 | 30.0 | 30.8 | done |
| TIE-GANs | 2024 | 28.1 | 30.8 | done |
| PINN-TIE | 2022 | 25.2 | 30.8 | done |
| TIE-DIC | 2010 | 25.0 | 30.8 | done |
| Phase gradient DIC | 2015 | 22.0 | 30.8 | done |
| Simple deconvolution | 2000 | 18.0 | 30.8 | done |
| Richardson-Lucy (PWM) | — | 15.6 | 30.8 | done |
| CARE (PWM) | — | 15.6 | 30.8 | done |
| DIC-Net (PWM) | — | 15.6 | 30.8 | done |
| precomputed_baseline (test) | — | 15.6 | 30.8 | done |
