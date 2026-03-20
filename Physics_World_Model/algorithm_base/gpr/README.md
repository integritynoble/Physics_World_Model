# Ground-Penetrating Radar (GPR) (`gpr`)

Category: Remote Sensing

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | RDA [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | SAR-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `gpr_dl` | GPR-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.gpr import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.gpr import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| PGCDM (Physics-Guided Diffusion) | 2024 | 30.1 | 30.3 | done |
| RTM (Reverse Time Migration) | 2000 | 25.0 | 30.3 | done |
| PSTM | 2005 | 22.0 | 30.3 | done |
| Kirchhoff migration | 2000 | 20.0 | 30.3 | done |
| RDA [proxy] (PWM) | — | 11.9 | 30.3 | done |
| SAR-DL [proxy] (PWM) | — | 11.9 | 30.3 | done |
| GPR-Net [proxy] (PWM) | — | 11.9 | 30.3 | done |
| Raw B-scan (noisy input) | 2021 | 11.2 | 30.3 | done |
| precomputed_baseline (test) | — | 10.9 | 30.3 | done |
