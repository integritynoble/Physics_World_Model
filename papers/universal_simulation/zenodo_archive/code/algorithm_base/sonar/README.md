# Sonar Imaging (`sonar`)

Category: Remote Sensing

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FISTA-L2 (DAS) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | SonarSR-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | Sonar-CNN [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.sonar import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.sonar import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SwinIR | 2025 | 36.1 | 33.0 | partial |
| MUSIC | 1986 | 27.0 | 33.0 | done |
| MVDR/Capon beamforming | 1969 | 25.0 | 33.0 | done |
| FISTA-L2 (DAS) [proxy] (PWM) | — | 16.0 | 33.0 | done |
| SonarSR-Net [proxy] (PWM) | — | 16.0 | 33.0 | done |
| Sonar-CNN [proxy] (PWM) | — | 16.0 | 33.0 | done |
| precomputed_baseline (test) | — | 15.0 | 33.0 | done |
| Matched Filter (sparse) | 2024 | 12.0 | 33.0 | done |
