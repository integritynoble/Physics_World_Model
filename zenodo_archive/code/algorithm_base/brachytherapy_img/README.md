# Brachytherapy Imaging (`brachytherapy_img`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `brachy_dl` | BrachyNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.brachytherapy_img import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.brachytherapy_img import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| RL-ARCNN (metal artifact reduction) | 2018 | 38.1 | 30.2 | partial |
| DL-Recon [proxy] (PWM) | — | 33.1 | 30.2 | done |
| BrachyNet [proxy] (PWM) | — | 33.1 | 30.2 | done |
| Monte Carlo dose | 2005 | 28.0 | 30.2 | done |
| precomputed_baseline (test) | — | 25.2 | 30.2 | done |
| FBP | 1971 | 25.0 | 30.2 | done |
