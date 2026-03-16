# PET/CT Fusion (`pet_ct`)

Category: Multi-Modal Fusion

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `petct_dl` | PET-CT-Fusion-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.pet_ct import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.pet_ct import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Attention U-Net + diffusion | 2025 | 35.9 | 29.8 | partial |
| TrUNET-MAPEM | 2023 | 33.7 | 29.8 | partial |
| OSEM + CT AC | 2000 | 28.0 | 29.8 | done |
| MLEM | 1982 | 25.0 | 29.8 | done |
| MLEM (low-count, 2 iter) | 1982 | 15.0 | 29.8 | done |
| Adjoint [proxy] (PWM) | — | 14.0 | 29.8 | done |
| PnP-ADMM [proxy] (PWM) | — | 14.0 | 29.8 | done |
| PET-CT-Fusion-Net [proxy] (PWM) | — | 14.0 | 29.8 | done |
| precomputed_baseline (test) | — | 13.0 | 29.8 | done |
