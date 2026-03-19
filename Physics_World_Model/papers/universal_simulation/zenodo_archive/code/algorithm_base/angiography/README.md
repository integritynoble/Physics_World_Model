# X-ray Angiography (`angiography`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP (DSA baseline) | `pwm_core.recon.ct_solvers.run_fbp` | No |  |
| `best_quality` | DSA-Net [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | VesselSegNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.angiography import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.angiography import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Maskless 2D-DSA (U-Net) | 2022 | 43.0 | 35.5 | partial |
| DSA subtraction (with motion) | 1980 | 30.0 | 35.5 | done |
| DSA (Digital Subtraction) | 1980 | 25.0 | 35.5 | done |
| Deep Decoupling Net (GAN+RDB) | 2024 | 23.7 | 35.5 | done |
| DSA-Net [proxy] (PWM) | — | 14.6 | 35.5 | done |
| VesselSegNet [proxy] (PWM) | — | 14.6 | 35.5 | done |
| precomputed_baseline (test) | — | 12.9 | 35.5 | done |
