# Cryo-EM Single Particle Analysis (`cryo_em`)

Category: Scientific Instrumentation
Carrier: Electron. DAG: M→P→D.

## Solvers (25 total: 11 classical + 14 deep learning)

### Classical / Iterative (11)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `traditional_cpu` | Wiener-CTF Correction | No | Penczek et al. 2010, Methods Enzymol. |
| `phase_flip` | Phase-Flip CTF Correction | No | Rosenthal & Henderson 2003, JMB |
| `back_projection` | Back-Projection | No | Radermacher 1988, J. Electron Microsc. Tech. |
| `sirt_3d` | SIRT (Simultaneous Iterative) | No | Gilbert 1972, J. Theor. Biol. |
| `landweber` | Landweber Iteration | No | Landweber 1951, Amer. J. Math. |
| `tikhonov` | Tikhonov Regularisation | No | Tikhonov 1963, Soviet Math. Doklady |
| `tv_admm` | Total Variation ADMM | No | Boyd et al. 2011, ADMM; Rudin-Osher-Fatemi 1992 TV |
| `pnp_admm_nlm` | PnP-ADMM (NLM denoiser) | No | Venkatakrishnan et al. 2013, GlobalSIP |
| `weighted_bp` | Weighted Back-Projection | No | Radermacher 1988; Harauz & van Heel 1986 |
| `cgls` | CGLS (Conjugate Gradient Least Squares) | No | Hestenes & Stiefel 1952, J. Res. NBS |
| `pnp_fista_nlm` | PnP-FISTA (NLM denoiser) | No | Beck & Teboulle 2009, SIAM J. Imaging Sci. |

### Deep Learning (14)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `best_quality` | RELION (PnP-PGD DRUNet) | Yes | Scheres 2012, JMB; Zivanov et al. 2018, eLife |
| `cryosparc` | CryoSPARC (PnP-PGD DRUNet) | Yes | Punjani et al. 2017, Nature Methods |
| `famous_dl` | CryoDRGN (PnP-PGD DRUNet) | Yes | Zhong et al. 2021, Nature Methods |
| `cryodrgn2` | CryoDRGN2 (PnP-HQS DRUNet) | Yes | Zhong et al. 2021, ICLR |
| `small_gpu` | CryoAI (DnCNN denoise) | Yes | Levy et al. 2022, NeurIPS |
| `deep_em_enhancer` | DeepEMenhancer (DRUNet denoise) | Yes | Sanchez-Garcia et al. 2021, Comms. Biol. |
| `topaz_denoise` | Topaz-Denoise (DRUNet denoise) | Yes | Bepler et al. 2020, Nature Comms. |
| `cryostar` | CryoSTAR (PnP-DRS DRUNet) | Yes | Guo et al. 2024, Nature Methods |
| `cryo_mamba` | CryoMamba (RED DRUNet) | Yes | Li et al. 2024, arXiv |
| `pnp_hqs_drunet` | PnP-HQS DRUNet | Yes | Zhang et al. 2017, CVPR (DnCNN/DRUNet) |
| `cryo_gan` | CryoGAN (PnP-PGD DRUNet) | Yes | Gupta et al. 2020, NeurIPS |
| `cryo_fire` | CryoFIRE (PnP-DRS DRUNet) | Yes | Zhong et al. 2023, ICLR |
| `cryo_former` | CryoFormer (PnP-PGD DRUNet) | Yes | CryoFormer 2024 |
| `cryo_foundation` | CryoFoundation (RED DRUNet) | Yes | CryoFoundation 2025 |

### Algorithms Not Yet Implemented

The following algorithms from the literature are documented but not yet implemented as solvers:

- **EMAN2** — single particle reconstruction (Tang et al., J. Struct. Biol. 2007)
- **Frealign** — Fourier-space refinement (Grigorieff 2007, J. Struct. Biol.)
- **cisTEM** — computational imaging for TEM (Grant et al. 2018, eLife)
- **3D Variability Analysis** — heterogeneity (Punjani & Fleet, J. Struct. Biol. 2021)
- **Neural volumes** — neural implicit representations (Mildenhall et al. 2020)
- **DynaMight** — dynamics modelling (Schwab et al. 2024)

## Usage

```python
# Import and run
from algorithm_base.cryo_em import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.cryo_em import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Topaz-Denoise | 2020 | 25.0 | 24.7 | done |
| DUAL (cryo-ET) | 2024 | 21.3 | 24.7 | done |
| DRA (denoising-recon) | 2024 | 20.2 | 24.7 | done |
| Adjoint [proxy] (PWM) | — | 20.2 | 24.7 | done |
| PnP-ADMM [proxy] (PWM) | — | 20.2 | 24.7 | done |
| CryoDRGN [proxy] (PWM) | — | 20.2 | 24.7 | done |
| cryoSPARC | 2017 | 20.0 | 24.7 | done |
| precomputed_wiener (test) | — | 19.2 | 24.7 | done |
| rl_ctf_20iter (test) | — | 19.2 | 24.7 | done |
| RELION | 2012 | 18.0 | 24.7 | done |
