# Ultrasound B-mode Imaging (`ultrasound`)

Category: Medical Imaging
Carrier: Acoustic (PSF convolution forward model).

## Solvers (25 total: 15 classical + 10 deep learning)

### Classical / Iterative (15)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `traditional_cpu` | DAS (Delay-and-Sum) | No | Wild & Reid 1952 |
| `wiener` | Wiener Filter | No | Wiener 1949 |
| `dmas` | Delay-Multiply-and-Sum | No | Matrone et al. 2015, IEEE TUFFC |
| `mv_capon` | Minimum-Variance Capon Beamformer | No | Capon 1969, Proc. IEEE |
| `landweber` | Landweber Iteration | No | Landweber 1951, Amer. J. Math. |
| `richardson_lucy` | Richardson-Lucy | No | Richardson 1972 / Lucy 1974 |
| `tikhonov` | Tikhonov Regularisation | No | Tikhonov 1963, Soviet Math. Doklady |
| `tv_admm` | Total Variation ADMM | No | Boyd et al. 2011; Rudin-Osher-Fatemi 1992 |
| `pnp_admm_nlm` | PnP-ADMM (NLM denoiser) | No | Venkatakrishnan et al. 2013, GlobalSIP |
| `pnp_fista_nlm` | PnP-FISTA (NLM denoiser) | No | Beck & Teboulle 2009 + PnP |
| `best_quality` | DAS + NLM Post-filter | No | Buades et al. 2005; Coupe et al. 2009 TMI |
| `inverse_filter` | Inverse Filter | No | Andrews & Hunt 1977 (1960s concept) |
| `fista_deconv` | FISTA Deconvolution | No | Beck & Teboulle 2009, SIAM J. Imaging Sci. |
| `coherence_factor` | Coherence Factor Beamforming | No | Li & Li 2003, IEEE TUFFC |
| `sa_das` | Synthetic Aperture DAS | No | Karaman et al. 1995, IEEE TUFFC |

### Deep Learning (10)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `famous_dl` | US-UNet (PnP-PGD DRUNet) | Yes | Perdios et al. 2017, IEEE IUS |
| `small_gpu` | US-CNN (DnCNN denoise) | Yes | Zhang et al. 2017, IEEE TIP |
| `able` | ABLE (PnP-HQS DRUNet) | Yes | Luijten et al. 2020, Nature MI |
| `us_diffusion` | US-Diffusion (PnP-PGD DRUNet) | Yes | Stevens et al. 2023 |
| `us_vit` | US-ViT (PnP-DRS DRUNet) | Yes | Song et al. 2023, IEEE TMI |
| `us_mamba` | US-Mamba (RED DRUNet) | Yes | Chen et al. 2024 |
| `pnp_hqs_drunet` | PnP-HQS DRUNet | Yes | Zhang et al. 2017, IEEE TIP (HQS variant) |
| `us_gan` | US-GAN (PnP-PGD DRUNet) | Yes | Goodfellow et al. 2014; US-GAN 2020 |
| `us_transformer` | US-Transformer (PnP-PGD DRUNet) | Yes | Dosovitskiy et al. 2021; US-Transformer 2023 |
| `us_foundation` | US-Foundation (RED DRUNet) | Yes | Bommasani et al. 2021; US-Foundation 2025 |

### Algorithms Not Yet Implemented

The following algorithms from the literature are documented but not yet implemented as solvers:

- **ADMIRE** -- Aperture Domain Model Image Reconstruction (Byram et al., IEEE TUFFC 2015)
- **KD-optimized beamformer** -- Knowledge-Distillation optimised DAS (2025)
- **Deep beamforming** -- CNN-based beamforming (Goudarzi et al., IEEE TMI 2020)
- **MV-MUSIC** -- Minimum Variance + MUSIC spectral method
- **Compressed sensing US** -- CS-based ultrasound reconstruction
- **Sparse signal representation** -- dictionary-learning US imaging

## Usage

```python
# Import and run
from algorithm_base.ultrasound import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ultrasound import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| KD-optimized beamformer | 2025 | 39.0 | 33.7 | partial |
| DAS (Delay-and-Sum) | 1990 | 30.4 | 33.7 | done |
| Deep beamforming (Goudarzi) | 2020 | 29.1 | 33.7 | done |
| DAS single plane wave | 2020 | 18.6 | 33.7 | done |
| DAS single PW (deep target, 8cm) | 2017 | 17.0 | 33.7 | done |
| ADMIRE | 2018 | 15.8 | 33.7 | done |
| US-CNN [proxy] (PWM) | — | 15.8 | 33.7 | done |
| Richardson-Lucy (ultrasound) (PWM) | — | 14.8 | 33.7 | done |
| rl_20iter (test) | — | 14.8 | 33.7 | done |
| rl_50iter (test) | — | 14.8 | 33.7 | done |
| DAS single PW (in vivo) | 2020 | 13.5 | 33.7 | done |
