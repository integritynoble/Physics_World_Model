# Magnetic Force Microscopy (MFM) (`mfm`)

Category: Scanning Probe Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes | Weigert et al. 2018 |
| `mfm_dl` | MFM-UNet | `pwm_core.recon.mfm_solvers.mfm_dl_recon` | Yes | Kim, M. et al. (2021) DL for magnetic force microscopy, npj Comput. Mater. 7:87 |

## Usage

```python
# Import and run
from algorithm_base.mfm import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.mfm import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Interval-BCS (AFM) | 2019 | 43.2 | 27.2 | gap |
| Richardson-Lucy (PWM) | — | 34.3 | 27.2 | partial |
| CARE (PWM) | — | 34.3 | 27.2 | partial |
| MFM-UNet (PWM) | — | 34.3 | 27.2 | partial |
| precomputed_baseline (test) | — | 34.3 | 27.2 | partial |
| Adaptive Median (AFM) | 2019 | 33.9 | 27.2 | partial |
| Wiener deconvolution | 1949 | 26.0 | 27.2 | done |
| Deconvolution | 2000 | 24.0 | 27.2 | done |
