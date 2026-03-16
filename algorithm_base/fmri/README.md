# Functional MRI (BOLD fMRI) (`fmri`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | SENSE (fMRI) | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |
| `best_quality` | fMRI-Transformer [proxy] | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |
| `famous_dl` | DeepBold [proxy] | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |

## Usage

```python
# Import and run
from algorithm_base.fmri import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.fmri import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| E2E-VarNet | 2021 | 41.4 | 12.3 | gap |
| CS-fMRI | 2010 | 32.0 | 12.3 | gap |
| Zero-filled IFFT | 2000 | 25.0 | 12.3 | gap |
| fMRI-Transformer [proxy] (PWM) | — | 9.9 | 12.3 | done |
| DeepBold [proxy] (PWM) | — | 9.9 | 12.3 | done |
| SENSE (fMRI) (PWM) | — | 4.9 | 12.3 | done |
| zero_filled (test) | — | 4.9 | 12.3 | done |
