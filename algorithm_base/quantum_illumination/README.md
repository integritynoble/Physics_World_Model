# Quantum Illumination (`quantum_illumination`)

Category: Quantum Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `qi_dl` | QI-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.quantum_illumination import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.quantum_illumination import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 23.5 | 27.2 | done |
| PnP-ADMM [proxy] (PWM) | — | 23.5 | 27.2 | done |
| QI-DL [proxy] (PWM) | — | 23.5 | 27.2 | done |
| precomputed_baseline (test) | — | 20.2 | 27.2 | done |
| Optimal receiver | 2008 | 15.0 | 27.2 | done |
| Photon counting (classical) | 2000 | 12.0 | 27.2 | done |
