# Shear-Wave Elastography (`elastography`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | SENSE (displacement field) | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |
| `best_quality` | MRE-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | NLSI-Solver [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.elastography import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.elastography import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CNN-LSTM | 2024 | 32.7 | 24.3 | partial |
| Direct inversion | 2001 | 24.0 | 24.3 | done |
| Phase gradient | 2000 | 22.0 | 24.3 | done |
| Raw displacement (no filtering) | 2000 | 14.0 | 24.3 | done |
| MRE-Net [proxy] (PWM) | — | 12.0 | 24.3 | done |
| NLSI-Solver [proxy] (PWM) | — | 12.0 | 24.3 | done |
| SENSE (displacement field) (PWM) | — | 11.0 | 24.3 | done |
| precomputed_baseline (test) | — | 11.0 | 24.3 | done |
