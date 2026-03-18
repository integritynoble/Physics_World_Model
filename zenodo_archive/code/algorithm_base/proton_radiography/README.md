# Proton Radiography (`proton_radiography`)

Category: Scientific Instrumentation

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP (proton radiography) | `pwm_core.recon.ct_solvers.run_fbp` | No |  |
| `best_quality` | ProtonRecon-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | FBP-Proton [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.proton_radiography import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.proton_radiography import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CNN proton portal imaging | 2024 | 39.1 | 32.3 | partial |
| cGAN synthetic CT | 2023 | 29.0 | 32.3 | done |
| DROP-TVS | 2013 | 28.0 | 32.3 | done |
| FBP (straight-line approx) | 2003 | 25.0 | 32.3 | done |
| MLP (Most Likely Path) | 2004 | 22.0 | 32.3 | done |
| ProtonRecon-Net [proxy] (PWM) | — | 13.0 | 32.3 | done |
| FBP-Proton [proxy] (PWM) | — | 13.0 | 32.3 | done |
| precomputed_baseline (test) | — | 12.0 | 32.3 | done |
