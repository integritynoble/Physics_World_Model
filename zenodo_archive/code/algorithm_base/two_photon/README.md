# Two-Photon / Multiphoton Microscopy (`two_photon`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy (2P) | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | 2P-Net (CARE) | `pwm_core.recon.two_photon_solvers.two_photon_care_recon` | Yes | Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090 |
| `famous_dl` | 2P-DeepInterp | `pwm_core.recon.two_photon_solvers.deep_interp_recon` | Yes | Lecoq, J. et al. (2021) Removing independent noise in systems neuroscience using DeepInterpolation, Nature Methods 18:1401 |

## Usage

```python
# Import and run
from algorithm_base.two_photon import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.two_photon import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| UNet-Att (self-supervised) | 2025 | 38.3 | 18.9 | gap |
| DeepCAD | 2021 | 35.0 | 18.9 | gap |
| 2P-Net (CARE) (PWM) | — | 33.8 | 18.9 | gap |
| 2P-DeepInterp (PWM) | — | 33.8 | 18.9 | gap |
| precomputed_baseline (test) | — | 33.8 | 18.9 | gap |
| rl_20iter (test) | — | 33.8 | 18.9 | gap |
| Richardson-Lucy | 1972 | 27.0 | 18.9 | partial |
