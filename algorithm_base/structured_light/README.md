# Structured-Light Depth Camera (`structured_light`)

Category: Depth Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FISTA-L2 (phase unwrap) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | SL-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | FTPD [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.structured_light import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.structured_light import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SFNet (fringe-to-phase) | 2024 | 38.0 | 38.1 | done |
| Phase-shifting (4-step) | 1984 | 35.0 | 38.1 | done |
| Gray code | 2003 | 25.0 | 38.1 | done |
| SL-Net [proxy] (PWM) | — | 13.0 | 38.1 | done |
| FTPD [proxy] (PWM) | — | 13.0 | 38.1 | done |
| precomputed_baseline (test) | — | 8.3 | 38.1 | done |
