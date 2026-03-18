# Positron Emission Tomography (PET) (`pet`)

Category: Medical Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | FBP (emission tomography) | `pwm_core.recon.ct_solvers.run_fbp` | No |  |
| `best_quality` | NeuroLF-PET | `pwm_core.recon.pet_solvers.neurolF_pet_recon` | Yes | Häggström, I. et al. (2019) DeepPET: DL for PET reconstruction, Med. Image Anal. 58 |
| `famous_dl` | PET-DL (U-Net) | `pwm_core.recon.pet_solvers.pet_unet_recon` | Yes | Gong, K. et al. (2019) PET image reconstruction with DL, IEEE TMI 38(9) |

## Usage

```python
# Import and run
from algorithm_base.pet import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.pet import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| SwinIR-PET | 2023 | 39.9 | 40.3 | done |
| DeepPET | 2019 | 34.7 | 40.3 | done |
| FBP (emission tomography) (PWM) | — | 33.1 | 40.3 | done |
| NeuroLF-PET (PWM) | — | 33.1 | 40.3 | done |
| PET-DL (U-Net) (PWM) | — | 33.1 | 40.3 | done |
| fbp_ramlak (test) | — | 33.1 | 40.3 | done |
| fbp_shepp_logan (test) | — | 33.1 | 40.3 | done |
| precomputed_fbp (test) | — | 33.1 | 40.3 | done |
| MAP-OSEM | 2001 | 32.0 | 40.3 | done |
| OSEM | 1994 | 30.0 | 40.3 | done |
| MLEM | 1982 | 28.0 | 40.3 | done |
