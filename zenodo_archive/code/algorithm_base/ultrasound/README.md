# Ultrasound B-mode Imaging (`ultrasound`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy (ultrasound) | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | US-UNet (DeepUS) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | US-CNN [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.ultrasound import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ultrasound import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| KD-optimized beamformer | 2025 | 39.0 | 33.7 | partial |
| DAS (Delay-and-Sum) | 1990 | 30.4 | 33.7 | done |
| Deep beamforming (Goudarzi) | 2020 | 29.1 | 33.7 | done |
| DAS single plane wave | 2020 | 18.6 | 33.7 | done |
| DAS single PW (deep target, 8cm) | 2017 | 17.0 | 33.7 | done |
| ADMIRE | 2018 | 15.8 | 33.7 | done |
| US-CNN [proxy] (PWM) | — | 15.8 | 33.7 | done |
| Richardson-Lucy (ultrasound) (PWM) | — | 14.8 | 33.7 | done |
| rl_20iter (test) | — | 14.8 | 33.7 | done |
| rl_50iter (test) | — | 14.8 | 33.7 | done |
| DAS single PW (in vivo) | 2020 | 13.5 | 33.7 | done |
