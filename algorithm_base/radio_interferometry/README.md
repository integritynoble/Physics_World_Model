# Radio Interferometry (VLBI) (`radio_interferometry`)

Category: Remote Sensing

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | RDA [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | SAR-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `ri_dl` | R2D2 (interferometry) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.radio_interferometry import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.radio_interferometry import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CASA tclean | 2007 | 28.0 | 22.4 | partial |
| MEM | 1984 | 27.0 | 22.4 | partial |
| CLEAN | 1974 | 25.0 | 22.4 | done |
| RDA [proxy] (PWM) | — | 24.5 | 22.4 | done |
| SAR-DL [proxy] (PWM) | — | 24.5 | 22.4 | done |
| R2D2 (interferometry) [proxy] (PWM) | — | 24.5 | 22.4 | done |
| precomputed_baseline (test) | — | 23.3 | 22.4 | done |
