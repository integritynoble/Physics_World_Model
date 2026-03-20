# Single Photon Emission CT (SPECT) (`spect`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP (emission tomography) | `pwm_core.recon.ct_solvers.run_fbp` | No |  |
| `best_quality` | SPECT-DL (OSEM+) | `pwm_core.recon.spect_solvers.spect_dl_recon` | Yes | Shiri, I. et al. (2020) Deep-JASC DL SPECT, Eur. J. Nucl. Med. Mol. Imaging |
| `famous_dl` | SPECT-UNet | `pwm_core.recon.spect_solvers.spect_unet_recon` | Yes | Kim, K. et al. (2018) Penalized PET reconstruction using DL, IEEE TMI 37(6) |

## Usage

```python
# Import and run
from algorithm_base.spect import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.spect import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| DIP-SPECT | 2020 | 33.3 | 33.3 | done |
| FBP (emission tomography) (PWM) | — | 30.0 | 33.3 | done |
| SPECT-DL (OSEM+) (PWM) | — | 30.0 | 33.3 | done |
| SPECT-UNet (PWM) | — | 30.0 | 33.3 | done |
| fbp_ramlak (test) | — | 30.0 | 33.3 | done |
| precomputed_fbp (test) | — | 30.0 | 33.3 | done |
| OSEM | 1994 | 28.5 | 33.3 | done |
| MLEM | 1982 | 26.0 | 33.3 | done |
