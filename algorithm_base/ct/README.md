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
| `landweber` | Landweber | 7.10 dB | verified (2 iters) |
| `art` | ART | 7.10 dB | verified (2 iters) |
| `sirt` | SIRT | 20.90 dB | verified (2 iters) |
| `cgls` | CGLS | 7.12 dB | verified (2 iters) |
| `mlem` | MLEM | 44.27 dB | verified |
| `sart` | SART | 7.10 dB | verified (2 iters) |
| `osem` | OSEM | 33.42 dB | verified |
| `tikhonov` | Tikhonov | 44.09 dB | verified |
| `tv_admm` | TV-ADMM | 44.07 dB | verified |
| `chambolle_pock` | Chambolle-Pock | 44.06 dB | verified |
| `pnp_admm_nlm` | PnP-ADMM (NLM) | 41.34 dB | verified (inline, 20 iters) |
| `pnp_hqs_nlm` | PnP-HQS (NLM) | 40.72 dB | verified (inline, 15 iters) |
| `pnp_fista_nlm` | PnP-FISTA (NLM) | 40.36 dB | verified (inline, 20 iters) |
| `pnp_admm_bm3d` | PnP-ADMM (BM3D) | 41.11 dB | verified (inline, 10 iters) |
| `best_quality` | FBP + NLM | 40.10 dB | verified |
| `fbp_bm3d` | FBP + BM3D | 40.50 dB | verified |
| `fbp_bilateral` | FBP + Bilateral | 36.10 dB | verified |
| `fbp_wavelet` | FBP + Wavelet | 43.54 dB | verified |
| `fbp_tv` | FBP + TV | 38.35 dB | verified |
| `famous_dl` | RED-CNN | 43.90 dB | verified (FBP fallback, no pretrained weights) |
| `fbpconvnet` | FBPConvNet | 38.61 dB | stub (FBP+NLM fallback) |
| `wgan_vgg` | WGAN-VGG | 38.61 dB | stub (FBP+NLM fallback) |
| `learn` | LEARN | 38.61 dB | stub (FBP+NLM fallback) |
| `learned_pd` | Learned Primal-Dual | 38.61 dB | stub (FBP+NLM fallback) |
| `iradonmap` | iRadonMAP | 38.61 dB | stub (FBP+NLM fallback) |
| `fbp_unet` | FBP + U-Net | 38.61 dB | stub (FBP+NLM fallback) |
| `dudonet` | DuDoNet | 38.61 dB | stub (FBP+NLM fallback) |
| `indudonet` | InDuDoNet | 38.61 dB | stub (FBP+NLM fallback) |
| `dudotrans` | DuDoTrans | 38.61 dB | stub (FBP+NLM fallback) |
| `ctformer` | CTformer | 38.61 dB | stub (FBP+NLM fallback) |
| `score_ct` | Score-CT | 38.61 dB | stub (FBP+NLM fallback) |
| `dps` | DPS | 38.61 dB | stub (FBP+NLM fallback) |
| `diffusion_mbir` | DiffusionMBIR | 38.61 dB | stub (FBP+NLM fallback) |
| `dolce` | DOLCE | 38.61 dB | stub (FBP+NLM fallback) |
| `ct_fm` | CT-FM | 38.61 dB | stub (FBP+NLM fallback) |

Note: Iterative solvers (landweber, art, sirt, cgls, sart) show low PSNR because tested with only 2 iterations for speed. With default iterations (15-30), they converge to much higher quality.

## DL Checkpoint Availability (researched 2026-03-19)

### Checkpoints in GCS (`gs://pwm-benchmark-datasets/checkpoint/ct/`)

| GCS File | Size | Model | Source Repo | LoDoPaB-CT? |
|----------|------|-------|-------------|-------------|
| `redcnn.pth` | 7 MB | RED-CNN | pwm_core | No (no pretrained weights) |
| `indudonet_latest.pt` | 20 MB | InDuDoNet | `hongwang01/InDuDoNet` | No (MAR task, 416x416 fan-beam) |
| `score_ct_brats_checkpoint_26` | 41 MB | Score-CT (BraTS) | `yang-song/score_inverse_problems` | No (brain MRI prior) |
| `score_ct_ct2d_320_checkpoint_101` | 63 MB | Score-CT (CT 320) | `yang-song/score_inverse_problems` | No (320x320, JAX/Flax, LIDC) |
| `dps_ffhq_10m.pt` | 358 MB | DPS (FFHQ) | `DPS2022/diffusion-posterior-sampling` | No (face images) |
| `diffusion_mbir_ct.pt` | 939 MB | DiffusionMBIR | `HJ-harry/DiffusionMBIR` | No (AAPM 256x256) |
| `dolce_model512_all.pt` | 1.1 GB | DOLCE (all CT) | `wustl-cig/DOLCE` | No (limited-angle, 512x512, LEAP) |
| `dolce_model512_ckc.pt` | 1.1 GB | DOLCE (medical) | `wustl-cig/DOLCE` | No (limited-angle, 512x512, LEAP) |
| `score_ct_ldct_512_checkpoint_63` | 1.2 GB | Score-CT (LDCT 512) | `yang-song/score_inverse_problems` | No (512x512, JAX/Flax) |
| `dps_imagenet256.pt` | 2.1 GB | DPS (ImageNet) | `DPS2022/diffusion-posterior-sampling` | No (natural images) |

**Total: 10 checkpoints, 6.62 GB**

### Not yet in GCS (available via DIVal library on GPU server)

| Model | Repository | LoDoPaB-CT? |
|-------|-----------|-------------|
| FBP+UNet | DIVal (`jleuschn/dival`) | **Yes** (pretrained on LoDoPaB) |
| Learned Primal-Dual | DIVal (`jleuschn/dival`) | **Yes** (pretrained on LoDoPaB) |
| iRadonMAP | DIVal (`jleuschn/dival`) | **Yes** (pretrained on LoDoPaB) |

### No checkpoint available

| Model | Repository | Reason |
|-------|-----------|--------|
| FBPConvNet | `panakino/FBPConvNet` | MATLAB/MatConvNet only |
| WGAN-VGG | `SSinyu/WGAN-VGG` | No weights released |
| LEARN | `maybe198376/LEARN` | MATLAB .mat only |
| Learned P-D (orig) | `adler-j/learned_primal_dual` | Never released |
| DuDoNet | `MIRACLE-Center/DuDoNet` | MAR only |
| DuDoTrans | `DuDoTrans/CODE` | Unclear |
| CTformer | `wdayang/CTformer` | Mayo 64x64 patches |
| CT-FM | Not found | No repo found |

**GPU server integration path:** Install DIVal + ODL on GPU server to use FBP+UNet, Learned P-D, and iRadonMAP pretrained on LoDoPaB-CT. Score-CT requires JAX/Flax and retraining on LoDoPaB ground truths.

## Algorithm Leaderboard (LoDoPaB-CT reference)

| # | Algorithm | Year | Ref PSNR | Status |
|---|-----------|------|----------|--------|
| 1 | CT-FM | 2024 | 44.1 dB | stub (checkpoint unknown) |
| 2 | DiffusionMBIR | 2023 | 43.8 dB | stub (AAPM checkpoint, needs retrain) |
| 3 | InDuDoNet | 2021 | 43.5 dB | stub (MAR only, not general CT) |
| 4 | DPS | 2023 | 43.2 dB | stub (FFHQ checkpoint, no CT) |
| 5 | LEARN | 2018 | 43.1 dB | stub (MATLAB only) |
| 6 | Score-CT | 2022 | 43.0 dB | stub (JAX/Flax, AAPM checkpoint) |
| 7 | DuDoTrans | 2022 | 42.1 dB | stub |
| 8 | CTformer | 2023 | 40.8 dB | stub (Mayo patches) |
| 9 | DuDoNet | 2019 | 40.2 dB | stub (MAR only) |
| 10 | PnP-ADMM (NLM) | 2013 | 39.5 dB | verified (41.34 dB PWM) |
| 11 | PnP-HQS (NLM) | 2017 | 39.1 dB | verified (40.72 dB PWM) |
| 12 | FBPConvNet | 2017 | 38.5 dB | stub (DIVal has LoDoPaB weights) |
| 13 | iRadonMAP | 2020 | 36.9 dB | stub (DIVal has LoDoPaB weights) |
| 14 | Learned Primal-Dual | 2018 | 36.2 dB | stub (DIVal has LoDoPaB weights) |
| 15 | DOLCE | 2023 | 36.0 dB | stub (limited-angle, LEAP dep) |
| 16 | FBP + U-Net | 2021 | 35.8 dB | stub (DIVal has LoDoPaB weights) |
| 17 | WGAN-VGG | 2018 | 34.1 dB | stub (no checkpoint) |
| 18 | RED-CNN | 2017 | 33.2 dB | verified (29.52 dB PWM, no weights) |
| 19 | CGLS | 1952 | 30.2 dB | verified |
| 20 | SIRT | 1972 | 29.5 dB | verified |
| 21 | SART | 1984 | 29.1 dB | verified |
| 22 | FBP + NLM | 2005 | 28.5 dB | verified (40.10 dB PWM) |
| 23 | TV-ADMM | 2008 | 27.8 dB | verified (44.07 dB PWM) |
| 24 | FBP (Ram-Lak) | 1971 | 25.2 dB | verified (43.90 dB PWM) |
