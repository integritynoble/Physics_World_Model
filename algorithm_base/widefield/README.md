# Widefield Fluorescence Microscopy (`widefield`)

Category: Microscopy
Carrier: Photon (incoherent fluorescence). Forward model: PSF convolution.

## Solvers (25 total: 13 classical + 12 deep learning)

### Classical / Iterative (13)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `traditional_cpu` | Richardson-Lucy Deconvolution | No | Richardson 1972 / Lucy 1974 |
| `wiener` | Wiener Filter | No | Wiener 1949, Extrapolation, Interpolation, and Smoothing |
| `gold` | Gold Deconvolution | No | Gold 1964, ANL Report 6984 |
| `jansson` | Jansson-van Cittert Iteration | No | van Cittert 1931, Zeitschrift f. Physik; Jansson 1970 |
| `landweber` | Landweber Iteration | No | Landweber 1951, Amer. J. Math. |
| `tikhonov` | Tikhonov Regularisation | No | Tikhonov 1963, Soviet Math. Doklady |
| `tv_deconv` | Total Variation Deconvolution | No | Rudin et al. 1992, Physica D |
| `rl_tv` | Richardson-Lucy with TV Regularisation | No | Dey et al. 2006, Microscopy Res. Tech. |
| `pnp_admm_nlm` | PnP-ADMM (NLM denoiser) | No | Venkatakrishnan et al. 2013, GlobalSIP |
| `pnp_fista_nlm` | PnP-FISTA (NLM denoiser) | No | Beck & Teboulle 2009, SIAM J. Imaging Sci. + PnP |
| `inverse_filter` | Inverse Filter | No | Direct Fourier division, 1960s |
| `agard` | Agard Constrained Iterative Deconvolution | No | Agard 1984, Ann. Rev. Biophys. Bioeng. |
| `regularized_rl` | Regularized Richardson-Lucy | No | Conchello 1998, JOSA A; Llacer & Nunez 1990 |

### Deep Learning (12)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `best_quality` | CARE (PnP-PGD DRUNet) | Yes | Weigert et al. 2018, Nature Methods |
| `famous_dl` | Noise2Void (PnP-PGD DRUNet) | Yes | Krull et al. 2019, CVPR |
| `small_gpu` | CSBDeep (DnCNN denoise) | Yes | Weigert et al. 2018, Nature Methods |
| `restormer` | Restormer (PnP-HQS DRUNet) | Yes | Zamir et al. 2022, CVPR |
| `wf_diffusion` | WF-Diffusion (PnP-PGD DRUNet) | Yes | Xie et al. 2023, arXiv |
| `deepcad_rt` | DeepCAD-RT (PnP-DRS DRUNet) | Yes | Li et al. 2023, Nature Methods |
| `wf_mamba` | WF-Mamba (RED DRUNet) | Yes | Wang et al. 2024, arXiv |
| `pnp_hqs_nlm_v2` | PnP-HQS (NLM v2) | No | Venkatakrishnan et al. 2013; HQS variant 2017 |
| `pnp_pgd_drunet` | PnP-PGD DRUNet | Yes | Zhang et al. 2017, PnP-PGD framework |
| `wf_gan` | WF-GAN (PnP-PGD DRUNet) | Yes | GAN-based widefield restoration, 2020 |
| `sr_resnet` | SRResNet (DnCNN denoise) | Yes | Ledig et al. 2017, CVPR |
| `wf_foundation` | WF-Foundation (RED DRUNet) | Yes | Foundation model for widefield, 2025 |

### Algorithms Not Yet Implemented

The following algorithms from the literature are documented but not yet implemented as solvers:

- **Blind Deconvolution** -- joint PSF + image estimation for widefield (Holmes 1992, JOSA A)
- **MAP-EM** -- Maximum A Posteriori Expectation-Maximisation (Conchello & McNally 1996)
- **Iteratively Constrained ML** -- constrained maximum likelihood (van Kempen & van Vliet 2000)
- **Good's Roughness** -- RL with Good's roughness penalty (Conchello 1998)
- **Multi-View Deconvolution** -- fusion of multiple views (Preibisch et al. 2010)
- **PURE-LET** -- Poisson unbiased risk estimation (Li et al. 2018)
- **Deconvolution-Lab2** -- GPU-accelerated deconvolution suite (Sage et al. 2017)
- **Content-Aware Compressed Sensing** -- CS for fluorescence (Wohlberg 2017)

## Usage

```python
# Import and run
from algorithm_base.widefield import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.widefield import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Restormer | 2022 | 35.5 | 47.8 | done |
| Noise2Void | 2019 | 31.0 | 47.8 | done |
| Wiener deconvolution | 1949 | 26.0 | 47.8 | done |
| precomputed_baseline (test) | -- | 25.0 | 47.8 | done |
| m-rBCR | 2023 | 24.9 | 47.8 | done |
| CARE | 2018 | 22.1 | 47.8 | done |
| Richardson-Lucy (20 iter) | 1972 | 13.4 | 47.8 | done |
