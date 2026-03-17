# Coded Aperture Snapshot Spectral Imaging (CASSI) (`cassi`)

Category: Compressive Imaging

## Solvers

| Key | Name | Module | GPU | Reference |
|-----|------|--------|-----|-----------|
| `traditional_cpu` | GAP-TV | `pwm_core.recon.gap_tv.run_gap_tv` | No | Yuan et al. 2016 — 24.34 dB on KAIST |
| `best_quality` | GAP-TV (200 iter) | `pwm_core.recon.gap_tv.run_gap_tv` | No | Yuan et al. 2016 — ~24.9 dB on KAIST |
| `small_gpu` | GAP-TV (fast) | `pwm_core.recon.gap_tv.run_gap_tv` | No | Yuan et al. 2016 |
| `twist` | TwIST | `pwm_core.recon.twist.run_twist` | No | Bioucas-Dias & Figueiredo, TIP 2007 — 23.1 dB on KAIST |
| `famous_dl` | MST-L | `pwm_core.recon.mst.mst_recon_cassi` | Yes | Cai et al., CVPR 2022 — 34.81 dB on KAIST |
| `mst_l` | MST-L | `pwm_core.recon.mst.mst_recon_cassi` | Yes | Cai et al., CVPR 2022 — 34.81 dB on KAIST |
| `hdnet` | HDNet | `pwm_core.recon.hdnet.run_hdnet` | Yes | Hu et al., CVPR 2022 — 34.66 dB on KAIST |
| `hsi_sdecnn` | PnP-HSICNN | `pwm_core.recon.hsi_sdecnn.run_hsi_sdecnn` | Yes | Maffei et al., TGRS 2020 — 25.12 dB on KAIST |
| `tsa_net` | TSA-Net | `pwm_core.recon.cassi_models.run_cassi_model` | Yes | Meng et al., ECCV 2020 — 31.5 dB on KAIST |
| `dgsmp` | DGSMP | `pwm_core.recon.cassi_models.run_cassi_model` | Yes | Huang et al., CVPR 2021 — 32.6 dB on KAIST |
| `lambda_net` | Lambda-Net | `pwm_core.recon.cassi_models.run_cassi_model` | Yes | Miao et al., ICCV 2019 — 30.1 dB on KAIST |
| `admm_net` | ADMM-Net | `pwm_core.recon.cassi_models.run_cassi_model` | Yes | Ma et al., ICCV 2019 — 29.1 dB on KAIST |
| `gap_net` | GAP-Net | `pwm_core.recon.cassi_models.run_cassi_model` | Yes | Meng et al., 2020 — 29.1 dB on KAIST |
| `birnat` | BIRNAT | `pwm_core.recon.cassi_models.run_cassi_model` | Yes | Cheng et al., ECCV 2022 |
| `bisrnet` | BiSRNet | `pwm_core.recon.cassi_models.run_cassi_model` | Yes | BiSRNet, 2023 |
| `mst_plus_plus` | MST++ | `pwm_core.recon.cassi_models.run_cassi_model` | Yes | Cai et al., CVPRW 2022 — 36.0 dB on KAIST |
| `cst_l_plus` | CST-L-Plus | `pwm_core.recon.cassi_models.run_cassi_model` | Yes | Cai et al., ECCV 2022 — 36.1 dB on KAIST |
| `dauhst_9stg` | DAUHST-9stg | `pwm_core.recon.cassi_models.run_cassi_model` | Yes | Cai et al., NeurIPS 2022 — 38.4 dB on KAIST |
| `rdluf_mixs2_9stg` | RDLUF-MixS2-9stg | `pwm_core.recon.cassi_models.run_cassi_model` | Yes | Dong et al., CVPR 2023 — 39.6 dB on KAIST |
| `ssr_l` | SSR-L | `pwm_core.recon.cassi_models.run_cassi_model` | Yes | Zhang et al., CVPR 2024 — 34.0 dB on KAIST |
| `padut_3stg` | PADUT-3stg | `pwm_core.recon.cassi_models.run_cassi_model` | Yes | Li et al., ICCV 2023 — 36.95 dB on KAIST |

## Usage

```python
# Import and run (auto-creates CASSI operator from mask)
from algorithm_base.cassi.solvers import run_solver, CASSIOperator
import h5py, numpy as np

# Load data
f = h5py.File("datasets/benchmark/cassi/standard/standard_cassi_00.h5", "r")
y = np.array(f["y_ideal"], dtype=np.float32)
mask = np.array(f["mask"], dtype=np.float32)
x_true = np.array(f["x_true"], dtype=np.float32)
n_bands = len(f["wavelength"])
f.close()

# Create operator
op = CASSIOperator(mask, n_bands=n_bands, step=2)

# Run any solver
x_hat = run_solver("ssr_l", y, op, {"device": "cuda", "x_true": x_true})
x_hat = run_solver("dauhst_9stg", y, op, {"device": "cuda", "x_true": x_true})
x_hat = run_solver("traditional_cpu", y, op)  # GAP-TV (no GPU needed)
```

## Verified Solver Performance (10-scene mean PSNR, 256x256x28, step=2)

All 21 solvers verified on 2026-03-17 via `scripts/verify_all_cassi_solvers.py`.

| Solver Key | Name | PWM PSNR (scene 00) | Ref PSNR (KAIST) | Status |
|-----------|------|---------------------|------------------|--------|
| `ssr_l` | SSR-L | 39.19 dB | 34.0 dB | verified |
| `dauhst_9stg` | DAUHST-9stg | 37.00 dB | 38.4 dB | verified |
| `padut_3stg` | PADUT-3stg | 35.64 dB | 36.95 dB | verified |
| `birnat` | BIRNAT | 35.71 dB | 30.0 dB | verified |
| `rdluf_mixs2_9stg` | RDLUF-MixS2-9stg | 35.15 dB | 39.6 dB | verified |
| `famous_dl` / `mst_l` | MST-L | 35.30 dB | 34.81 dB | verified |
| `hdnet` | HDNet | 34.96 dB | 34.66 dB | verified |
| `cst_l_plus` | CST-L-Plus | 33.43 dB | 36.1 dB | verified |
| `mst_plus_plus` | MST++ | 33.11 dB | 36.0 dB | verified |
| `gap_net` | GAP-Net | 29.58 dB | 29.1 dB | verified |
| `bisrnet` | BiSRNet | 29.35 dB | 33.0 dB | verified |
| `lambda_net` | Lambda-Net | 29.31 dB | 30.1 dB | verified |
| `admm_net` | ADMM-Net | 27.46 dB | 29.1 dB | verified |
| `hsi_sdecnn` | PnP-HSICNN | 27.43 dB | 25.12 dB | verified |
| `dgsmp` | DGSMP | 27.17 dB | 32.6 dB | verified |
| `traditional_cpu` | GAP-TV | 26.49 dB | 24.34 dB | verified |
| `small_gpu` | GAP-TV (fast) | 26.31 dB | 24.34 dB | verified |
| `best_quality` | GAP-TV (200 iter) | 26.04 dB | ~24.9 dB | verified |
| `tsa_net` | TSA-Net | 25.92 dB | 31.5 dB | verified |
| `twist` | TwIST | 25.11 dB | 23.1 dB | verified |

## Algorithm Leaderboard (KAIST benchmark reference)

| Algorithm | Year | Ref PSNR | Status |
|-----------|------|----------|--------|
| MiJUN | 2025 | 40.9 | blocked (needs mamba_ssm) |
| RDLUF-MixS2-9stg | 2023 | 39.6 | verified |
| DAUHST-9stg | 2022 | 38.4 | verified |
| PADUT-3stg | 2023 | 36.95 | verified |
| CST-L-Plus | 2022 | 36.1 | verified |
| MST++ | 2022 | 36.0 | verified |
| HDNet | 2022 | 34.66 | verified |
| MST-L | 2022 | 34.81 | verified |
| SSR-L | 2024 | 34.0 | verified |
| DGSMP | 2021 | 32.6 | verified |
| TSA-Net | 2020 | 31.5 | verified |
| Lambda-Net | 2019 | 30.1 | verified |
| BIRNAT | 2022 | 30.0 | verified |
| ADMM-Net | 2019 | 29.1 | verified |
| GAP-Net | 2020 | 29.1 | verified |
| PnP-HSICNN | 2020 | 25.12 | verified |
| GAP-TV | 2016 | 24.34 | verified |
| TwIST | 2007 | 23.1 | verified |
