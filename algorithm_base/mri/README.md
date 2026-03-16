# Magnetic Resonance Imaging (MRI) (`mri`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | Zero-Filled IFFT | `pwm_core.recon.mri_solvers.run_zero_filled` | No |  |
| `best_quality` | CS-MRI (Wavelet) | `pwm_core.recon.mri_solvers.run_cs_mri` | No | Lustig et al. 2007, MRM |
| `famous_dl` | MoDL | `pwm_core.recon.modl.run_modl` | No | Aggarwal et al. 2019, IEEE TMI |
| `small_gpu` | MoDL (5 unrolls) | `pwm_core.recon.modl.run_modl` | No |  |
| `sense` | SENSE | `pwm_core.recon.mri_solvers.run_sense` | No | Pruessmann et al., MRM 1999 |

## Usage

```python
# Import and run
from algorithm_base.mri import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.mri import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| PromptMR | 2023 | 41.5 | 16.0 | gap |
| E2E-VarNet | 2020 | 40.5 | 16.0 | gap |
| ReconFormer | 2023 | 40.1 | 16.0 | gap |
| PromptMR+ | 2024 | 39.9 | 16.0 | gap |
| HUMUS-Net | 2022 | 37.3 | 16.0 | gap |
| U-Net | 2018 | 36.0 | 16.0 | gap |
| GRAPPA | 2002 | 34.0 | 16.0 | gap |
| CS-MRI (SparseMRI) | 2007 | 33.0 | 16.0 | gap |
| Zero-filled IFFT | 2000 | 28.0 | 16.0 | gap |
| E2E-VarNet (16x) | 2024 | 23.2 | 16.0 | partial |
| Zero-filled (32x accel) | 2018 | 15.0 | 16.0 | done |
| CS-MRI (Wavelet) (PWM) | — | 13.4 | 16.0 | done |
| MoDL (PWM) | — | 13.4 | 16.0 | done |
| MoDL (5 unrolls) (PWM) | — | 13.4 | 16.0 | done |
| zero_filled (test) | — | 13.4 | 16.0 | done |
| cs_mri_wavelet (test) | — | 13.4 | 16.0 | done |
| sense (test) | — | 13.4 | 16.0 | done |
