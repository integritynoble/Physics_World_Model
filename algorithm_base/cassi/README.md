# Coded Aperture Snapshot Spectral Imaging (CASSI) (`cassi`)

Category: Compressive Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | GAP-TV | `pwm_core.recon.gap_tv.run_gap_tv` | No | Yuan et al. 2016 |
| `best_quality` | GAP-TV (guided) | `pwm_core.recon.gap_tv.run_gap_tv` | No | Yuan et al. 2016 |
| `famous_dl` | GAP-TV (fast) | `pwm_core.recon.gap_tv.run_gap_tv` | No |  |
| `small_gpu` | GAP-TV (small) | `pwm_core.recon.gap_tv.run_gap_tv` | No |  |
| `mst_l` | MST-L | `pwm_core.recon.mst.mst_recon_cassi` | No | Cai et al., CVPR 2022 |
| `hdnet` | HDNet | `pwm_core.recon.hdnet.run_hdnet` | No | Hu et al., CVPR 2022 |
| `hsi_sdecnn` | HSI-SDeCNN | `pwm_core.recon.hsi_sdecnn.run_hsi_sdecnn` | No | Maffei et al., TGRS 2020 |

## Usage

```python
# Import and run
from algorithm_base.cassi import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.cassi import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| MiJUN | 2025 | 40.9 | 10.1 | gap |
| RDLUF-MixS2 | 2022 | 39.6 | 10.1 | gap |
| PADUT-L | 2023 | 38.9 | 10.1 | gap |
| DAUHST-9stg | 2022 | 38.4 | 10.1 | gap |
| CST-L-Plus | 2022 | 36.1 | 10.1 | gap |
| MST++ | 2022 | 36.0 | 10.1 | gap |
| HDNet | 2022 | 35.0 | 10.1 | gap |
| MST-L | 2022 | 34.9 | 10.1 | gap |
| PADUT | 2023 | 34.8 | 10.1 | gap |
| SSR-L | 2023 | 34.0 | 10.1 | gap |
| DGSMP | 2021 | 32.6 | 10.1 | gap |
| TSA-Net | 2020 | 31.5 | 10.1 | gap |
| λ-Net | 2020 | 30.1 | 10.1 | gap |
| ADMM-Net | 2019 | 29.1 | 10.1 | gap |
| GAP-TV (guided) (PWM) | — | 26.2 | 10.1 | gap |
| GAP-TV (fast) (PWM) | — | 26.2 | 10.1 | gap |
| GAP-TV (small) (PWM) | — | 26.2 | 10.1 | gap |
| GAP-TV | 2016 | 24.4 | 10.1 | gap |
| TwIST | 2007 | 23.1 | 10.1 | gap |
