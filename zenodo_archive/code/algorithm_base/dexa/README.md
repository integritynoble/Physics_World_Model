# Dual-Energy X-ray Absorptiometry (DEXA) (`dexa`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FISTA-L2 (dual-energy) | `pwm_core.recon.classical.run_fista_l2` | No |  |
| `best_quality` | DXA-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | DEXA-UNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.dexa import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.dexa import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DL bone density estimation | 2022 | 32.0 | 49.7 | done |
| Dual-energy decomposition | 1987 | 28.0 | 49.7 | done |
| Bone decomposition baseline | 2020 | 19.7 | 49.7 | done |
| DXA-Net [proxy] (PWM) | — | 11.7 | 49.7 | done |
| DEXA-UNet [proxy] (PWM) | — | 11.7 | 49.7 | done |
| FISTA-L2 (dual-energy) (PWM) | — | 10.7 | 49.7 | done |
| precomputed_baseline (test) | — | 10.7 | 49.7 | done |
