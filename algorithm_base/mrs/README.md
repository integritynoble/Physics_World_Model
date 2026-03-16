# MR Spectroscopy (MRS) (`mrs`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | SENSE (spectroscopy) | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |
| `best_quality` | MRS-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | HLSVD-MRS [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.mrs import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.mrs import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DDPM-MRSI (2x SR) | 2025 | 29.7 | 22.2 | partial |
| LCModel | 1993 | 28.0 | 22.2 | partial |
| HLSVD | 2002 | 22.0 | 22.2 | done |
| MRS-Net [proxy] (PWM) | — | 13.0 | 22.2 | done |
| SENSE (spectroscopy) (PWM) | — | 11.0 | 22.2 | done |
| precomputed_baseline (test) | — | 11.0 | 22.2 | done |
