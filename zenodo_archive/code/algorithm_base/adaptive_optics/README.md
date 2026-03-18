# Adaptive Optics (AO) Imaging (`adaptive_optics`)

Category: Broader Experimental Science

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `deep_ao` | Deep-AO [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.adaptive_optics import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.adaptive_optics import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 101.0 | 43.1 | gap |
| PnP-ADMM [proxy] (PWM) | — | 101.0 | 43.1 | gap |
| Deep-AO [proxy] (PWM) | — | 101.0 | 43.1 | gap |
| precomputed_baseline (test) | — | 100.0 | 43.1 | gap |
| cGAN wavefront | 2020 | 31.0 | 43.1 | done |
| Phase diversity | 1982 | 26.0 | 43.1 | done |
| Shack-Hartmann WFS | 1971 | 22.0 | 43.1 | done |
