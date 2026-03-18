# Acoustic Emission Testing (AE) (`acoustic_emission`)

Category: Broader Experimental Science

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Adjoint [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | PnP-ADMM [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `dl_localizer` | DeepAE-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.acoustic_emission import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.acoustic_emission import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| CNN Beamformer (1 source) | 2023 | 39.4 | 31.1 | partial |
| CNN Beamformer (3 sources) | 2023 | 32.3 | 31.1 | done |
| MUSIC localization | 1986 | 22.0 | 31.1 | done |
| Adjoint [proxy] (PWM) | — | 21.6 | 31.1 | done |
| PnP-ADMM [proxy] (PWM) | — | 21.6 | 31.1 | done |
| DeepAE-Net [proxy] (PWM) | — | 21.6 | 31.1 | done |
| precomputed_baseline (test) | — | 20.2 | 31.1 | done |
| AIC picker | 2000 | 20.0 | 31.1 | done |
