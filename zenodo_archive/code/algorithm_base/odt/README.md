# Optical Diffraction Tomography (ODT) (`odt`)

Category: Coherent Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `odt_dl` | ODT-Net (PhaseNet) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.odt import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.odt import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 30.5 | 42.5 | done |
| PnP-ADMM [proxy] (PWM) | — | 30.5 | 42.5 | done |
| ODT-Net (PhaseNet) [proxy] (PWM) | — | 30.5 | 42.5 | done |
| precomputed_baseline (test) | — | 27.2 | 42.5 | done |
| Rytov approximation | 2000 | 25.0 | 42.5 | done |
| Born approximation | 2000 | 22.0 | 42.5 | done |
