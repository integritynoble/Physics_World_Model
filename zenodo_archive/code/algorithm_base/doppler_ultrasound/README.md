# Doppler Ultrasound (`doppler_ultrasound`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Back-Projection (Doppler) | `pwm_core.recon.photoacoustic_solver.run_photoacoustic` | No |  |
| `best_quality` | UDoppler-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | Doppler CFAR [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.doppler_ultrasound import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.doppler_ultrasound import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DL Doppler | 2020 | 30.0 | 25.6 | partial |
| 3D-Res-UNet (95% compression) | 2022 | 26.7 | 25.6 | done |
| Autocorrelation | 1985 | 22.0 | 25.6 | done |
| Conventional SVD (90% compression) | 2022 | 19.5 | 25.6 | done |
| UDoppler-Net [proxy] (PWM) | — | 18.6 | 25.6 | done |
| Doppler CFAR [proxy] (PWM) | — | 18.6 | 25.6 | done |
| Wall filter (highpass) | 1985 | 18.0 | 25.6 | done |
| Back-Projection (Doppler) (PWM) | — | 17.6 | 25.6 | done |
| autocorrelation_estimator (test) | — | 17.6 | 25.6 | done |
| clutter_filtered (test) | — | 17.6 | 25.6 | done |
| precomputed_baseline (test) | — | 17.6 | 25.6 | done |
| Conventional SVD (95% compression) | 2022 | 17.4 | 25.6 | done |
