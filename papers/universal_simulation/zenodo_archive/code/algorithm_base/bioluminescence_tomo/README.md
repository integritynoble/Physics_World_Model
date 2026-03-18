# Bioluminescence Tomography (BLT) (`bioluminescence_tomo`)

Category: Broader Experimental Science

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `blt_dl` | BLT-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.bioluminescence_tomo import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.bioluminescence_tomo import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| L1-regularized BLT | 2010 | 22.0 | 37.4 | done |
| Diffusion-model inversion | 2005 | 18.0 | 37.4 | done |
| Adjoint [proxy] (PWM) | — | 14.3 | 37.4 | done |
| PnP-ADMM [proxy] (PWM) | — | 14.3 | 37.4 | done |
| BLT-Net [proxy] (PWM) | — | 14.3 | 37.4 | done |
| precomputed_baseline (test) | — | 13.3 | 37.4 | done |
| Direct mapping | 2000 | 12.0 | 37.4 | done |
