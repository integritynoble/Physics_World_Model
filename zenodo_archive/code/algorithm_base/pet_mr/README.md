# PET/MR Fusion (`pet_mr`)

Category: Multi-Modal Fusion

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `petmr_dl` | PET-MR-DeepJoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.pet_mr import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.pet_mr import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Brain DL PET/MR | 2024 | 42.0 | 29.6 | gap |
| MRAC-based reconstruction | 2010 | 26.0 | 29.6 | done |
| No-AC reconstruction | 2010 | 15.0 | 29.6 | done |
| Adjoint [proxy] (PWM) | — | 14.5 | 29.6 | done |
| PnP-ADMM [proxy] (PWM) | — | 14.5 | 29.6 | done |
| PET-MR-DeepJoint [proxy] (PWM) | — | 14.5 | 29.6 | done |
| No-AC (1/10 counts) | 2010 | 13.0 | 29.6 | done |
| precomputed_baseline (test) | — | 12.5 | 29.6 | done |
