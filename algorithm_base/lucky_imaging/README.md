# Lucky Imaging (`lucky_imaging`)

Category: Astronomy & Space Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `lucky_dl` | Lucky-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.lucky_imaging import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.lucky_imaging import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 32.7 | 27.1 | partial |
| PnP-ADMM [proxy] (PWM) | — | 32.7 | 27.1 | partial |
| Lucky-DL [proxy] (PWM) | — | 32.7 | 27.1 | partial |
| precomputed_baseline (test) | — | 30.0 | 27.1 | done |
| DiffIR2VR-Zero | 2025 | 27.8 | 27.1 | done |
| RVRT+ | 2025 | 26.5 | 27.1 | done |
| Drizzle | 2002 | 26.0 | 27.1 | done |
| Shift-and-add | 2000 | 22.0 | 27.1 | done |
