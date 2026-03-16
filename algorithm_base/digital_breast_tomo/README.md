# Digital Breast Tomosynthesis (DBT) (`digital_breast_tomo`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `dbt_dl` | DBT-DL [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.digital_breast_tomo import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.digital_breast_tomo import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SART | 1984 | 30.0 | 24.1 | partial |
| TV-regularized MLEM | 2010 | 28.0 | 24.1 | partial |
| FBP | 1971 | 25.0 | 24.1 | done |
| DL-Recon [proxy] (PWM) | — | 10.5 | 24.1 | done |
| DBT-DL [proxy] (PWM) | — | 10.5 | 24.1 | done |
| precomputed_baseline (test) | — | 8.8 | 24.1 | done |
