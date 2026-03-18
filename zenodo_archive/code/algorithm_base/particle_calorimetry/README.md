# Particle Calorimetry (`particle_calorimetry`)

Category: Broader Experimental Science

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `cal_dl` | CaloDiffusion [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.particle_calorimetry import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.particle_calorimetry import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 37.7 | 20.4 | gap |
| PnP-ADMM [proxy] (PWM) | — | 37.7 | 20.4 | gap |
| CaloDiffusion [proxy] (PWM) | — | 37.7 | 20.4 | gap |
| precomputed_baseline (test) | — | 36.7 | 20.4 | gap |
| Pandora PFA | 2014 | 22.0 | 20.4 | done |
| Clustering algorithms | 2000 | 20.0 | 20.4 | done |
