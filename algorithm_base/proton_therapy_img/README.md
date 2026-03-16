# Proton Therapy Imaging (`proton_therapy_img`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `best_quality` | DL-Recon [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `proton_therapy_dl` | ProtonTherapy-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.proton_therapy_img import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.proton_therapy_img import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Residual GAN (PPI-to-DRR) | 2024 | 39.1 | 30.7 | partial |
| CycleGAN (CBCT-to-sCT) | 2024 | 34.1 | 30.7 | partial |
| Proton CT DL | 2022 | 32.0 | 30.7 | done |
| DL-Recon [proxy] (PWM) | — | 31.2 | 30.7 | done |
| FBP | 1971 | 28.0 | 30.7 | done |
| precomputed_baseline (test) | — | 26.6 | 30.7 | done |
