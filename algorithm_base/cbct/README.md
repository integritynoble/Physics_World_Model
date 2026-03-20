# Cone-Beam Computed Tomography (CBCT) (`cbct`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FDK / FBP | `pwm_core.recon.ct_solvers.run_fbp` | No |  |
| `best_quality` | FDK-DL | `pwm_core.recon.cbct_solvers.fdk_dl_recon` | Yes | Chen, H. et al. (2017) Low-dose CT with residual encoder-decoder CNN, IEEE TMI |
| `famous_dl` | CBCT-UNet | `pwm_core.recon.cbct_solvers.cbct_unet_recon` | Yes | Jin, K.H. et al. (2017) Deep convolutional network for inverse problems, IEEE TIP |

## Usage

```python
# Import and run
from algorithm_base.cbct import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.cbct import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| FBPConvNet | 2017 | 36.5 | 15.1 | gap |
| FACT | 2022 | 33.8 | 15.1 | gap |
| SART | 1984 | 32.0 | 15.1 | gap |
| FDK | 1984 | 28.0 | 15.1 | gap |
| FDK (8 views) | 1984 | 16.6 | 15.1 | done |
| FDK (6 views) | 1984 | 15.3 | 15.1 | done |
| FDK-DL (PWM) | — | 15.2 | 15.1 | done |
| CBCT-UNet (PWM) | — | 15.2 | 15.1 | done |
| fbp_ramlak (test) | — | 15.2 | 15.1 | done |
| fbp_shepp_logan (test) | — | 15.2 | 15.1 | done |
