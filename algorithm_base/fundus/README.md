# Fundus Camera (`fundus`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | RETFound | `pwm_core.recon.fundus_solvers.retfound_recon` | Yes | Zhou, Y. et al. (2023) RETFound: Foundation model for retinal imaging, Nature 622:156 |
| `famous_dl` | DR-Grade-Net | `pwm_core.recon.fundus_solvers.dr_grade_net_recon` | Yes | Gulshan, V. et al. (2016) DL for DR detection in retinal fundus, JAMA 316(22) |

## Usage

```python
# Import and run
from algorithm_base.fundus import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.fundus import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| RETFound (PWM) | — | 35.9 | 36.8 | done |
| DR-Grade-Net (PWM) | — | 35.9 | 36.8 | done |
| rl_20iter (test) | — | 35.9 | 36.8 | done |
| rl_50iter (test) | — | 35.9 | 36.8 | done |
| precomputed_wiener (test) | — | 35.9 | 36.8 | done |
| Richardson-Lucy | 1972 | 30.0 | 36.8 | done |
| PCE-Net | 2023 | 29.9 | 36.8 | done |
| GFE-Net | 2023 | 29.7 | 36.8 | done |
| Cofe-Net | 2022 | 24.9 | 36.8 | done |
