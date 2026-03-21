# Lensless (Diffuser Camera) Imaging (`lensless`)

Category: Computational Photography
Carrier: Incoherent (PSF convolution). DAG: M->P->D.

## Solvers (26 total: 13 classical + 13 deep learning)

### Classical / Iterative (13)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `inverse_filter` | Inverse Filter | No | Classical Fourier optics, direct spectral inversion, 1960s |
| `wiener` | Wiener Deconvolution | No | Wiener 1949 |
| `tikhonov` | Tikhonov Regularisation | No | Tikhonov 1963 |
| `constrained_ls` | Constrained Least Squares | No | Hunt, IEEE Trans. Computers, 1973 |
| `traditional_cpu` | Richardson-Lucy Deconvolution | No | Richardson 1972; Lucy 1974 |
| `landweber` | Landweber Iteration | No | Landweber 1951 |
| `gradient_descent` | Gradient Descent Deconvolution | No | Standard iterative gradient descent, 1980s |
| `fista_deconv` | FISTA Deconvolution | No | Beck & Teboulle, SIAM J. Imaging Sciences, 2009 |
| `admm_l1_wavelet` | ADMM-L1 (Wavelet) | No | Boyd et al., Found. Trends ML, 2011 |
| `tv_admm` | TV-ADMM Deconvolution | No | Boyd et al. 2011; Chambolle 2004 |
| `admm_tv` | ADMM-TV (Lensless) | No | Antipa et al., Optica, 2018 |
| `pnp_admm_nlm` | PnP-ADMM (NLM) | No | Venkatakrishnan et al., IEEE GlobalSIP, 2013 |
| `pnp_hqs_nlm` | PnP-HQS (NLM) | No | Zhang et al., CVPR, 2017 |

### Deep Learning (13)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `pnp_pgd_drunet` | PnP-PGD (DRUNet) | Yes | Zhang et al., IEEE TPAMI, 2017/2022 |
| `best_quality` | FlatNet | Yes | Khan et al., IEEE TPAMI, 2020 |
| `famous_dl` | Le-ADMM-U | Yes | Monakhova et al., IEEE TPAMI, 2022 |
| `small_gpu` | FlatNet-Lite | Yes | Khan et al., IEEE TPAMI, 2020 |
| `phlatcam` | PhlatCam | Yes | Boominathan et al., IEEE TPAMI / ICCP, 2020 |
| `unrolled_admm` | Unrolled ADMM | Yes | Deep unrolled ADMM for lensless, 2020 |
| `l3fnet` | L3Fnet | Yes | Tan et al., IEEE TMM, 2023 |
| `diffuser_dm` | DiffuserDM | Yes | Diffusion model for diffuser cameras, 2023 |
| `digicam_net` | DigiCam-Net | Yes | CNN-based digital camera reconstruction, 2023 |
| `lensless_former` | LenslessFormer | Yes | Cao et al., CVPR, 2024 |
| `lens_mamba` | LensMamba | Yes | Mamba-based lensless reconstruction, 2024 |
| `lensless_diffusion` | Lensless-Diffusion | Yes | Diffusion model for lensless imaging, 2024 |
| `lensless_foundation` | Lensless-Foundation | Yes | Foundation model for lensless imaging, 2025 |

### Algorithms Not Yet Implemented

The following algorithms from the literature are documented but not yet implemented as solvers:

- **MWDN** -- Multi-Wiener Deconvolution Network (Zeng et al., Optica 2021)
- **LensNet** -- end-to-end CNN for lensless (2025)
- **SLNet** -- Self-supervised Lensless Network (Zeng et al. 2023)
- **Mask-FlowNet** -- optical flow for coded-aperture lensless (Zheng et al. 2023)
- **Privacy-preserving lensless** -- privacy-aware reconstruction (Hinojosa et al. 2022)
- **Depth-from-diffuser** -- depth estimation from lensless captures (Wu et al. 2019)
- **Video lensless** -- temporal reconstruction for lensless video (Pan et al. 2023)

## Usage

```python
# Import and run
from algorithm_base.lensless import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.lensless import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| LensNet | 2025 | 27.5 | 25.2 | done |
| MWDN | 2023 | 25.7 | 25.2 | done |
| FlatNet | 2022 | 21.2 | 25.2 | done |
| ADMM | 2000 | 12.8 | 25.2 | done |
| FlatNet-Lite (PWM) | — | 11.9 | 25.2 | done |
| wiener_deconv (test) | — | 11.9 | 25.2 | done |
| Wiener deconvolution | 2025 | 7.3 | 25.2 | done |
