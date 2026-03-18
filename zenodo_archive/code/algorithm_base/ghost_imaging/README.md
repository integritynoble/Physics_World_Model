# Ghost Imaging (`ghost_imaging`)

Category: Quantum Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `gi_dl` | GI-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.ghost_imaging import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ghost_imaging import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Orthogonal GI (2D-DCT) | 2025 | 30.0 | 27.2 | done |
| DGI-Net | 2021 | 28.0 | 27.2 | done |
| Bio-inspired self-attention | 2025 | 24.5 | 27.2 | done |
| CS-GI | 2013 | 22.0 | 27.2 | done |
| DeepGhost (autoencoder) | 2020 | 19.9 | 27.2 | done |
| Differential GI | 2010 | 18.0 | 27.2 | done |
| Correlation imaging | 2002 | 15.0 | 27.2 | done |
| Raw correlation (5% sampling) | 2002 | 10.0 | 27.2 | done |
| Correlation GI (natural, 128x128) | 2020 | 9.5 | 27.2 | done |
| Adjoint [proxy] (PWM) | — | 8.7 | 27.2 | done |
| PnP-ADMM [proxy] (PWM) | — | 8.7 | 27.2 | done |
| GI-Net [proxy] (PWM) | — | 8.7 | 27.2 | done |
| Traditional GI (3000 measurements) | 2021 | 7.2 | 27.2 | done |
| precomputed_baseline (test) | — | 6.6 | 27.2 | done |
