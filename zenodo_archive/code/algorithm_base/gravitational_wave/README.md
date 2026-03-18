# Gravitational Wave Detection (`gravitational_wave`)

Category: Broader Experimental Science

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `matched_filter_dl` | GW-DL (PyCBC-ML) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.gravitational_wave import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.gravitational_wave import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 101.0 | 37.7 | gap |
| PnP-ADMM [proxy] (PWM) | — | 101.0 | 37.7 | gap |
| GW-DL (PyCBC-ML) [proxy] (PWM) | — | 101.0 | 37.7 | gap |
| precomputed_baseline (test) | — | 100.0 | 37.7 | gap |
| BayesWave | 2015 | 25.0 | 37.7 | done |
| cWaveNet | 2020 | 22.0 | 37.7 | done |
| Matched filtering | 2000 | 20.0 | 37.7 | done |
