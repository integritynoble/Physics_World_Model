# Talbot-Lau X-ray Grating Interferometry (`talbot_lau`)

Category: Coherent Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `talbot_dl` | Talbot-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.talbot_lau import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.talbot_lau import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 34.3 | 25.8 | partial |
| PnP-ADMM [proxy] (PWM) | — | 34.3 | 25.8 | partial |
| Talbot-Net [proxy] (PWM) | — | 34.3 | 25.8 | partial |
| precomputed_baseline (test) | — | 28.9 | 25.8 | partial |
| Phase-stepping | 2006 | 28.0 | 25.8 | done |
| Fourier analysis | 2006 | 25.0 | 25.8 | done |
