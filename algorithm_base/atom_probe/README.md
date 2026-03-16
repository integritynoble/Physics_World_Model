# Atom Probe Tomography (APT) (`atom_probe`)

Category: Scientific Instrumentation

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `apt_dl` | APT-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.atom_probe import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.atom_probe import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 42.1 | 26.0 | gap |
| PnP-ADMM [proxy] (PWM) | — | 42.1 | 26.0 | gap |
| APT-Net [proxy] (PWM) | — | 42.1 | 26.0 | gap |
| precomputed_baseline (test) | — | 41.1 | 26.0 | gap |
| ML trajectory correction | 2022 | 24.0 | 26.0 | done |
| Voltage reconstruction | 2000 | 20.0 | 26.0 | done |
