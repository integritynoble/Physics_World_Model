# X-ray Crystallography (`xray_crystallography`)

Category: Scientific Instrumentation

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `xtal_dl` | AlphaFold-SF [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.xray_crystallography import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.xray_crystallography import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SHELXD | 2010 | 28.0 | 25.8 | done |
| Adjoint [proxy] (PWM) | — | 23.4 | 25.8 | done |
| PnP-ADMM [proxy] (PWM) | — | 23.4 | 25.8 | done |
| AlphaFold-SF [proxy] (PWM) | — | 23.4 | 25.8 | done |
| precomputed_baseline (test) | — | 22.4 | 25.8 | done |
| Direct methods | 1953 | 22.0 | 25.8 | done |
