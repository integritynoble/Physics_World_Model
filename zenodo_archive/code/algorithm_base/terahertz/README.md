# Terahertz Imaging (THz) (`terahertz`)

Category: Industrial Inspection

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `thz_dl` | THz-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.terahertz import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.terahertz import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Adjoint [proxy] (PWM) | — | 47.9 | 35.2 | gap |
| PnP-ADMM [proxy] (PWM) | — | 47.9 | 35.2 | gap |
| THz-Net [proxy] (PWM) | — | 47.9 | 35.2 | gap |
| precomputed_baseline (test) | — | 37.1 | 35.2 | done |
| J-Net (real THz) | 2023 | 32.5 | 35.2 | done |
| EARDB | 2023 | 31.3 | 35.2 | done |
| TDS deconvolution | 2000 | 22.0 | 35.2 | done |
