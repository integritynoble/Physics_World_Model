# MR Fingerprinting (MRF) (`mr_fingerprinting`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `mrf_dl` | MRF-Net [proxy] | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |

## Usage

```python
# Import and run
from algorithm_base.mr_fingerprinting import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.mr_fingerprinting import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| MRF-Mixer (T2 map) | 2025 | 35.9 | 23.0 | gap |
| MRF-Mixer (T1 map) | 2025 | 33.5 | 23.0 | gap |
| GAST-Mamba (T1 map) | 2025 | 33.1 | 23.0 | gap |
| MANTIS | 2019 | 30.0 | 23.0 | partial |
| Dictionary matching | 2013 | 25.0 | 23.0 | done |
| FBP [proxy] (PWM) | — | 13.0 | 23.0 | done |
| DL-Recon [proxy] (PWM) | — | 13.0 | 23.0 | done |
| precomputed_baseline (test) | — | 11.0 | 23.0 | done |
