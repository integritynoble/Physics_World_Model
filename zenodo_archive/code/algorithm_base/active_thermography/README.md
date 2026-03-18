# Active Thermography (IR) (`active_thermography`)

Category: Industrial Inspection

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `pulsed_phase_tv` | Pulsed-Phase TV [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.active_thermography import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.active_thermography import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| TESR (Transformer) | 2024 | 46.2 | 35.1 | gap |
| RCAN | 2024 | 45.9 | 35.1 | gap |
| EDSR | 2024 | 45.3 | 35.1 | gap |
| SRCNN | 2024 | 42.9 | 35.1 | partial |
| Bicubic baseline | 2024 | 42.1 | 35.1 | partial |
| Pulsed phase thermography | 1996 | 25.0 | 35.1 | done |
| Adjoint [proxy] (PWM) | — | 8.2 | 35.1 | done |
| PnP-ADMM [proxy] (PWM) | — | 8.2 | 35.1 | done |
| precomputed_baseline (test) | — | 7.2 | 35.1 | done |
