# Coded Aperture Compressive Temporal Imaging (CACTI) (`cacti`)

Category: Compressive Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | GAP-TV | `pwm_core.recon.gap_tv.run_gap_tv` | No | Yuan et al. 2016 |
| `best_quality` | EfficientSCI | `pwm_core.recon.efficientsci.run_efficientsci` | No | Wang et al. CVPR 2023 |
| `famous_dl` | ELP-Unfolding | `pwm_core.recon.elp_unfolding.run_elp_unfolding` | No | Yang et al. ECCV 2022 |
| `small_gpu` | EfficientSCI-T | `pwm_core.recon.efficientsci.run_efficientsci` | No |  |
| `pnp_ffdnet` | PnP-FFDNet | `pwm_core.recon.cacti_solvers.pnp_ffdnet_cacti` | No | Yuan et al., CVPR 2020 |
| `hisvit9` | HiSViT-9 | `pwm_core.recon.cacti_solvers.hisvit_cacti` | Yes | Chen et al., ICCV 2023 |
| `hisvit13` | HiSViT-13 | `pwm_core.recon.cacti_solvers.hisvit_cacti` | Yes | Chen et al., ECCV 2024 |

## Usage

```python
# Import and run
from algorithm_base.cacti import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.cacti import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| HiSViT-13 | 2024 | 37.3 | 13.2 | gap |
| CTM-SCI | 2024 | 36.5 | 13.2 | gap |
| DUN-3DUnet | 2022 | 35.3 | 13.2 | gap |
| HiSViT | 2023 | 34.5 | 13.2 | gap |
| EfficientSCI | 2023 | 34.3 | 13.2 | gap |
| STFormer | 2022 | 33.9 | 13.2 | gap |
| ELP-Unfolding | 2022 | 33.1 | 13.2 | gap |
| BIRNAT | 2022 | 32.7 | 13.2 | gap |
| RevSCI-Net | 2021 | 31.4 | 13.2 | gap |
| MetaSCI | 2021 | 30.1 | 13.2 | gap |
| PnP-FFDNet | 2020 | 28.7 | 13.2 | gap |
| DeSCI | 2019 | 27.1 | 13.2 | gap |
| GAP-TV | 2016 | 26.7 | 13.2 | gap |
| GAP-TV (Traffic scene) | 2016 | 20.9 | 13.2 | partial |
| EfficientSCI-T (PWM) | — | 19.8 | 13.2 | partial |
| mask_division_baseline (test) | — | 19.8 | 13.2 | partial |
| gap_tv (test) | — | 19.8 | 13.2 | partial |
