# Diffusion MRI (DTI) (`diffusion_mri`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | SENSE (WLS tensor fit) | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |
| `best_quality` | q-DL (qDiffusion) [proxy] | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |
| `famous_dl` | SHORE-Net [proxy] | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |

## Usage

```python
# Import and run
from algorithm_base.diffusion_mri import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.diffusion_mri import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| q-DL | 2016 | 34.0 | 24.7 | partial |
| MPR-ViT (ADC maps) | 2024 | 31.0 | 24.7 | partial |
| Zero-filled IFFT | 2000 | 25.0 | 24.7 | done |
| Zero-filled (high b-value) | 2000 | 15.0 | 24.7 | done |
| SHORE-Net [proxy] (PWM) | — | 13.0 | 24.7 | done |
| Zero-filled (R=4, multi-b) | 2023 | 12.2 | 24.7 | done |
| Zero-filled (R=6, multi-b) | 2023 | 12.0 | 24.7 | done |
| SENSE (WLS tensor fit) (PWM) | — | 11.3 | 24.7 | done |
| zero_filled (test) | — | 11.3 | 24.7 | done |
