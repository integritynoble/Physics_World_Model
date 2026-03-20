# Electrical Impedance Tomography (EIT) (`impedance_tomo`)

Category: Broader Experimental Science

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `eit_dl` | EIT-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.impedance_tomo import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.impedance_tomo import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SA-HFL | 2023 | 31.0 | 29.9 | done |
| EIDORS-Net | 2020 | 26.0 | 29.9 | done |
| TV-ADMM | 2010 | 22.0 | 29.9 | done |
| Linear backprojection | 1990 | 22.0 | 29.9 | done |
| Newton one-step | 2005 | 20.0 | 29.9 | done |
| D-bar method | 2000 | 18.0 | 29.9 | done |
| Adjoint [proxy] (PWM) | — | 15.9 | 29.9 | done |
| PnP-ADMM [proxy] (PWM) | — | 15.9 | 29.9 | done |
| EIT-Net [proxy] (PWM) | — | 15.9 | 29.9 | done |
| TPINV (Tikhonov Pseudoinverse) | 2023 | 12.9 | 29.9 | done |
| precomputed_baseline (test) | — | 12.6 | 29.9 | done |
| LBP (Linear Back Projection) | 2023 | 12.4 | 29.9 | done |
