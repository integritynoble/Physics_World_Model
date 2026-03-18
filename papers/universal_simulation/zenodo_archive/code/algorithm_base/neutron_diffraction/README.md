# Neutron Diffraction (`neutron_diffraction`)

Category: Scientific Instrumentation

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `nd_dl` | NeutronDiff-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.neutron_diffraction import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.neutron_diffraction import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Rietveld refinement | 1969 | 25.0 | 27.2 | done |
| Le Bail fitting | 1988 | 22.0 | 27.2 | done |
| Adjoint [proxy] (PWM) | — | 10.3 | 27.2 | done |
| PnP-ADMM [proxy] (PWM) | — | 10.3 | 27.2 | done |
| NeutronDiff-Net [proxy] (PWM) | — | 10.3 | 27.2 | done |
| precomputed_baseline (test) | — | 8.8 | 27.2 | done |
