# X-ray Computed Tomography (CT) (`ct`)

Category: Medical Imaging

## Dataset

**LoDoPaB-CT** — real clinical chest CT from LIDC/IDRI database.
- Source: Leuschner et al., Scientific Data 2021 (doi:10.1038/s41597-021-00893-z)
- Zenodo: https://zenodo.org/records/3384092
- License: CC BY 4.0
- Image size: 362x362, parallel beam, 1000 angles, 512 detectors
- 20 standard samples from test split

## Solvers (41 total)

### Classical Analytical (1956–1974)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `traditional_cpu` | FBP (Ram-Lak) | No | Ramachandran & Lakshminarayanan 1971 |
| `fbp_shepp_logan` | FBP (Shepp-Logan) | No | Shepp & Logan, IEEE TNS 1974 |
| `fbp_cosine` | FBP (Cosine) | No | Standard windowed FBP |
| `fbp_hamming` | FBP (Hamming) | No | Hamming-windowed FBP |
| `fbp_hann` | FBP (Hann) | No | Hann-windowed FBP |

### Classical Iterative (1951–1994)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `landweber` | Landweber | No | Landweber, Am J Math 1951 |
| `art` | ART | No | Gordon, Bender & Herman 1970 |
| `sirt` | SIRT | No | Gilbert 1972 |
| `cgls` | CGLS | No | Hestenes & Stiefel 1952 |
| `mlem` | MLEM | No | Shepp & Vardi, IEEE TMI 1982 |
| `sart` | SART | No | Andersen & Kak 1984 |
| `osem` | OSEM | No | Hudson & Larkin, IEEE TMI 1994 |

### Regularization-based (1963–2011)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `tikhonov` | Tikhonov | No | Tikhonov 1963 |
| `tv_admm` | TV-ADMM | No | Sidky & Pan 2008 |
| `chambolle_pock` | Chambolle-Pock | No | Chambolle & Pock 2011 |

### Plug-and-Play (2013–2017)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `pnp_admm_nlm` | PnP-ADMM (NLM) | No | Venkatakrishnan et al. 2013 |
| `pnp_hqs_nlm` | PnP-HQS (NLM) | No | Zhang et al. 2017 |
| `pnp_fista_nlm` | PnP-FISTA (NLM) | No | Beck & Teboulle 2009 + PnP |
| `pnp_admm_bm3d` | PnP-ADMM (BM3D) | No | Venkatakrishnan et al. 2013 + BM3D |

### FBP + Post-processing

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `best_quality` | FBP + NLM | No | Buades et al. 2005 |
| `fbp_bm3d` | FBP + BM3D | No | Dabov et al. 2007 |
| `fbp_bilateral` | FBP + Bilateral | No | Tomasi & Manduchi 1998 |
| `fbp_wavelet` | FBP + Wavelet | No | Donoho 1995 |
| `fbp_tv` | FBP + TV | No | Rudin, Osher & Fatemi 1992 |

### Deep Learning (2017–2023)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `famous_dl` / `small_gpu` | RED-CNN | No | Chen et al., IEEE TMI 2017 |
| `fbpconvnet` | FBPConvNet | Yes | Jin et al., TIP 2017 |
| `wgan_vgg` | WGAN-VGG | Yes | Yang et al., IEEE TMI 2018 |
| `learn` | LEARN | Yes | Chen et al., IEEE TMI 2018 |
| `learned_pd` | Learned Primal-Dual | Yes | Adler & Öktem 2018 |
| `iradonmap` | iRadonMAP | Yes | He et al., MICCAI 2020 |
| `fbp_unet` | FBP + U-Net | Yes | Ronneberger et al. 2015 |
| `dudonet` | DuDoNet | Yes | Lin et al., CVPR 2019 |
| `indudonet` | InDuDoNet | Yes | Song et al., MICCAI 2021 |
| `dudotrans` | DuDoTrans | Yes | Wang et al., MICCAI 2022 |
| `ctformer` | CTformer | Yes | Wang et al., IEEE TMI 2023 |

### Diffusion / Score-based (2022–2024)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `score_ct` | Score-CT | Yes | Song et al., ICLR 2022 |
| `dps` | DPS | Yes | Chung et al., ICML 2023 |
| `diffusion_mbir` | DiffusionMBIR | Yes | Chung & Ye, NeurIPS 2023 |
| `dolce` | DOLCE | Yes | Liu et al. 2023 |
| `ct_fm` | CT-FM | Yes | Denker et al. 2024 |

## Usage

```python
from algorithm_base.ct.solvers import run_solver, SOLVERS

# Run any solver by key
x_hat = run_solver("traditional_cpu", sinogram, operator, cfg)

# List all solvers
from algorithm_base.ct.solvers import list_solvers
for key, spec in list_solvers():
    print(f"{key}: {spec['name']}")
```

## Verified Solver Performance (LoDoPaB-CT, 362x362, 1000 angles)

| Solver Key | Name | PWM PSNR | Status |
|-----------|------|----------|--------|
| `traditional_cpu` | FBP (Ram-Lak) | 43.90 dB | verified |
| `fbp_shepp_logan` | FBP (Shepp-Logan) | 41.65 dB | verified |
| `fbp_cosine` | FBP (Cosine) | 38.27 dB | verified |
| `fbp_hamming` | FBP (Hamming) | 36.93 dB | verified |
| `fbp_hann` | FBP (Hann) | 36.50 dB | verified |
| `sirt` | SIRT | 20.90 dB | verified (2 iters) |
| `cgls` | CGLS | 43.90 dB | verified (2 iters) |
| `mlem` | MLEM | 44.27 dB | verified (2 iters) |
| `tikhonov` | Tikhonov | 44.09 dB | verified (2 iters) |
| `chambolle_pock` | Chambolle-Pock | 44.06 dB | verified (2 iters) |
| `tv_admm` | TV-ADMM | 44.07 dB | verified (2 iters) |
| `best_quality` | FBP + NLM | 40.10 dB | verified |
| `fbp_bm3d` | FBP + BM3D | 40.10 dB | verified |
| `fbp_bilateral` | FBP + Bilateral | 36.10 dB | verified |
| `fbp_tv` | FBP + TV | 38.35 dB | verified |
| `famous_dl` | RED-CNN | 29.52 dB | verified (no pretrained weights) |

## Algorithm Leaderboard (LoDoPaB-CT reference)

| # | Algorithm | Year | Ref PSNR | Status |
|---|-----------|------|----------|--------|
| 1 | CT-FM | 2024 | 44.1 dB | registered |
| 2 | DiffusionMBIR | 2023 | 43.8 dB | registered |
| 3 | InDuDoNet | 2021 | 43.5 dB | registered |
| 4 | DPS | 2023 | 43.2 dB | registered |
| 5 | LEARN | 2018 | 43.1 dB | registered |
| 6 | Score-CT | 2022 | 43.0 dB | registered |
| 7 | DuDoTrans | 2022 | 42.1 dB | registered |
| 8 | CTformer | 2023 | 40.8 dB | registered |
| 9 | DuDoNet | 2019 | 40.2 dB | registered |
| 10 | PnP-ADMM | 2013 | 39.5 dB | verified |
| 11 | PnP-HQS | 2017 | 39.1 dB | verified |
| 12 | FBPConvNet | 2017 | 38.5 dB | registered |
| 13 | iRadonMAP | 2020 | 36.9 dB | registered |
| 14 | Learned Primal-Dual | 2018 | 36.2 dB | registered |
| 15 | DOLCE | 2023 | 36.0 dB | registered |
| 16 | FBP + U-Net | 2021 | 35.8 dB | registered |
| 17 | WGAN-VGG | 2018 | 34.1 dB | registered |
| 18 | RED-CNN | 2017 | 33.2 dB | verified |
| 19 | CGLS | 1952 | 30.2 dB | verified |
| 20 | SIRT | 1972 | 29.5 dB | verified |
| 21 | SART | 1984 | 29.1 dB | verified |
| 22 | FBP + NLM | 2005 | 28.5 dB | verified |
| 23 | TV-ADMM | 2008 | 27.8 dB | verified |
| 24 | FBP (Ram-Lak) | 1971 | 25.2 dB | verified |
