# Mammography (`mammography`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP (mammography) | `pwm_core.recon.ct_solvers.run_fbp` | No |  |
| `best_quality` | MammoNet (GatorTron) [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |
| `famous_dl` | Mammo-ResNet [proxy] | `pwm_core.recon.richardson_lucy.run_richardson_lucy` | No | Richardson 1972, JOSA |

## Usage

```python
# Import and run
from algorithm_base.mammography import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.mammography import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DeepTFormer | 2025 | 39.4 | 31.2 | partial |
| RED-CNN | 2017 | 35.0 | 31.2 | partial |
| BM3D | 2007 | 32.0 | 31.2 | done |
| FBP | 1971 | 30.0 | 31.2 | done |
| NLM denoising | 2005 | 26.0 | 31.2 | done |
| MammoNet (GatorTron) [proxy] (PWM) | — | 21.9 | 31.2 | done |
| Mammo-ResNet [proxy] (PWM) | — | 21.9 | 31.2 | done |
| precomputed_recon (test) | — | 20.9 | 31.2 | done |
