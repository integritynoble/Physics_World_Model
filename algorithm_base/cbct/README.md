# Cone-Beam Computed Tomography (CBCT) (`cbct`)

Category: Medical Imaging
Carrier: X-ray. DAG: M→P→D.

## Solvers (30 total: 20 classical + 10 deep learning)

### Classical / Iterative (20)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `traditional_cpu` | FDK Ram-Lak | No | Feldkamp, Davis & Kress 1984 |
| `fdk_shepp_logan` | FDK Shepp-Logan | No | Shepp & Logan 1974 |
| `fdk_hamming` | FDK Hamming | No | Feldkamp, Davis & Kress 1984 |
| `fdk_hann` | FDK Hann | No | Feldkamp, Davis & Kress 1984 |
| `landweber` | Landweber Iteration | No | Landweber 1951 |
| `art` | Algebraic Reconstruction Technique (ART) | No | Gordon, Bender & Herman 1970 |
| `sirt` | Simultaneous Iterative Reconstruction (SIRT) | No | Gilbert 1972 |
| `cgls` | Conjugate Gradient Least Squares (CGLS) | No | Hestenes & Stiefel 1952 |
| `sart` | Simultaneous ART (SART) | No | Andersen & Kak 1984 |
| `mlem` | ML-EM | No | Shepp & Vardi 1982 |
| `osem` | Ordered Subsets EM (OS-EM) | No | Hudson & Larkin 1994 |
| `tikhonov` | Tikhonov Regularization | No | Tikhonov 1963 |
| `tv_admm` | TV-ADMM | No | Sidky, Kao & Pan 2008 |
| `chambolle_pock` | Chambolle-Pock Primal-Dual | No | Chambolle & Pock 2011 |
| `pnp_admm_nlm` | PnP-ADMM with NLM | No | Venkatakrishnan et al. 2013 |
| `pnp_fista_nlm` | PnP-FISTA with NLM | No | Beck & Teboulle 2009 + PnP |
| `best_quality` | FDK + NLM Post-Processing | No | Buades, Coll & Morel 2005 |
| `fbp` | Filtered Back-Projection (FBP) | No | Ramachandran & Lakshminarayanan 1971 |
| `lsqr` | LSQR Iterative Solver | No | Paige & Saunders 1982 |
| `gradient_descent` | Gradient Descent | No | Natterer 1986 |

### Deep Learning (10)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `famous_dl` | FDK-DL (DRUNet) | Yes | Chen et al. 2017 |
| `small_gpu` | CBCT-UNet (DnCNN) | Yes | Jin et al. 2017 |
| `cbct_diffusion` | CBCT Diffusion (DRUNet) | Yes | Chung et al. 2023 |
| `cbct_naf` | CBCT Neural Attenuation Fields (DRUNet) | Yes | Zha et al. 2024 |
| `cbct_mamba` | CBCT-Mamba (DRUNet) | Yes | Wang et al. 2024 |
| `pnp_hqs_drunet` | PnP-HQS DRUNet | Yes | Romano, Elad & Milanfar 2017 |
| `cbct_gan` | CBCT-GAN (DRUNet) | Yes | Jiang et al. 2019 |
| `cbct_transformer` | CBCT-Transformer (DRUNet) | Yes | Wang et al. 2022 |
| `cbct_nerf` | CBCT-NeRF (DRUNet) | Yes | Zha et al. 2023 |
| `cbct_foundation` | CBCT-Foundation (RED-DRUNet) | Yes | Li et al. 2025 |

### Algorithms Not Yet Implemented

The following algorithms from the literature are documented but not yet implemented as solvers:

- **FBPConvNet** — Learned FBP post-processing (Jin et al., IEEE TIP 2017)
- **FACT** — Fast Adaptive CT reconstruction (2022)
- **LEARN** — Learned Experts' Assessment-based Reconstruction Network (Chen et al., IEEE TMI 2018)
- **iCT-Net** — Iterative CT Network (Li et al., IEEE TMI 2020)
- **DOLCE** — Diffusion Posterior Sampling for CBCT (Liu et al., 2024)
- **DuDoNet** — Dual-Domain Network for CT Metal Artifact Reduction (Lin et al., CVPR 2019)
- **RegFormer** — Regularization Transformer for CT (2023)
- **Score-based diffusion** — Score-based generative models for CT (Song et al., NeurIPS 2021)

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
