# Photon-Counting Spectral CT (`spectral_ct`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `spectral_ct_dl` | SpectralCT-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.spectral_ct import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.spectral_ct import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| D3QN | 2024 | 37.4 | 7.4 | gap |
| Butterfly-Net | 2022 | 34.0 | 7.4 | gap |
| ADMM-TV | 2010 | 30.0 | 7.4 | gap |
| Material decomposition | 2003 | 28.0 | 7.4 | gap |
| FBP per bin (lowest energy) | 2024 | 27.0 | 7.4 | gap |
| FBP (30 sparse views) | 2025 | 15.5 | 7.4 | partial |
| DL-Recon [proxy] (PWM) | — | 13.3 | 7.4 | partial |
| SpectralCT-Net [proxy] (PWM) | — | 13.3 | 7.4 | partial |
| precomputed_baseline (test) | — | 12.3 | 7.4 | partial |
