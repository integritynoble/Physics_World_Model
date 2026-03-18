# Correlative Light-Electron Microscopy (CLEM) (`clem`)

Category: Multi-Modal Fusion

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `clem_dl` | CLEM-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.clem import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.clem import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 39.7 | 26.0 | gap |
| PnP-ADMM [proxy] (PWM) | — | 39.7 | 26.0 | gap |
| CLEM-Net [proxy] (PWM) | — | 39.7 | 26.0 | gap |
| precomputed_baseline (test) | — | 28.1 | 26.0 | done |
| VoxelMorph registration | 2019 | 26.0 | 26.0 | done |
| Landmark registration | 2000 | 22.0 | 26.0 | done |
