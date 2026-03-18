# Low-Dose Widefield Microscopy (`widefield_lowdose`)

Category: Microscopy

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | BM3D + RL | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No |  |
| `best_quality` | CARE | `pwm_core.recon.care_unet.run_care` | Yes |  |
| `famous_dl` | Noise2Void | `pwm_core.recon.noise2void.noise2void_denoise` | No | Krull et al. CVPR 2019 |
| `small_gpu` | Noise2Void | `pwm_core.recon.noise2void.noise2void_denoise` | No |  |

## Usage

```python
# Import and run
from algorithm_base.widefield_lowdose import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.widefield_lowdose import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| BM3D + RL (PWM) | — | 29.0 | 35.6 | done |
| CARE (PWM) | — | 29.0 | 35.6 | done |
| precomputed_baseline (test) | — | 29.0 | 35.6 | done |
| Noise2Void | 2019 | 26.0 | 35.6 | done |
| Richardson-Lucy | 1972 | 20.0 | 35.6 | done |
