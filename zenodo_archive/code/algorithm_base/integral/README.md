# Integral Photography (`integral`)

Category: Computational Optics

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Depth Estimation | `pwm_core.recon.integral_solver.run_integral` | No |  |
| `best_quality` | DIBR [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | EPINet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `small_gpu` | EPINet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.integral import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.integral import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DIBR [proxy] (PWM) | — | 44.3 | 28.7 | gap |
| EPINet [proxy] (PWM) | — | 44.3 | 28.7 | gap |
| Depth Estimation (PWM) | — | 41.1 | 28.7 | gap |
| precomputed_baseline (test) | — | 41.1 | 28.7 | gap |
| Drizzle (IFS) | 2003 | 25.0 | 28.7 | done |
| PCA sky subtraction | 2012 | 22.0 | 28.7 | done |
