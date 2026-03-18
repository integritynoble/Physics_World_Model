# MR Angiography (MRA) (`mra`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `mra_dl` | MRA-VesselNet [proxy] | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |

## Usage

```python
# Import and run
from algorithm_base.mra import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.mra import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| 3D CNN SR | 2025 | 36.8 | 19.2 | gap |
| CS-MRA | 2010 | 30.0 | 19.2 | gap |
| Zero-filled (R=7-11) | 2024 | 25.8 | 19.2 | partial |
| Zero-filled IFFT | 2000 | 25.0 | 19.2 | partial |
| Zero-filled (16x accel) | 2026 | 25.0 | 19.2 | partial |
| FBP [proxy] (PWM) | — | 18.1 | 19.2 | done |
| DL-Recon [proxy] (PWM) | — | 18.1 | 19.2 | done |
| MRA-VesselNet [proxy] (PWM) | — | 18.1 | 19.2 | done |
| precomputed_baseline (test) | — | 14.7 | 19.2 | done |
