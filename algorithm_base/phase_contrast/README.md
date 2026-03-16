# Phase Contrast Microscopy (`phase_contrast`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `pc_dl` | PhaseNet | `pwm_core.recon.phase_contrast_solvers.phase_net_recon` | Yes | Rivenson, Y. et al. (2018) Phase recovery with DL, Light: Sci. & Appl. 7:17141 |

## Usage

```python
# Import and run
from algorithm_base.phase_contrast import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.phase_contrast import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Richardson-Lucy (PWM) | — | 45.6 | 28.0 | gap |
| CARE (PWM) | — | 45.6 | 28.0 | gap |
| PhaseNet (PWM) | — | 45.6 | 28.0 | gap |
| precomputed_baseline (test) | — | 45.6 | 28.0 | gap |
| GAN (self-attention) | 2024 | 38.3 | 28.0 | gap |
| Fourier ptychography | 2013 | 32.0 | 28.0 | partial |
| DL flat-fielding QPC | 2024 | 29.1 | 28.0 | done |
| TIE (Transport of Intensity) | 2001 | 28.0 | 28.0 | done |
