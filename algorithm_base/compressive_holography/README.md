# Compressive Digital Holography (`compressive_holography`)

Category: Coherent Imaging / Compressive Sensing

## Forward Model

Multi-depth holographic imaging: a 3D object (multiple depth planes) is encoded
into a single 2D hologram via Fresnel propagation and interference.

```
y = sum_k Re{ R* P(z_k) x_k }
```

where P(z_k) is Fresnel propagation to depth z_k, R is the reference wave.
Simplified PSF model: y = conv(x, psf) + noise.

## Solvers (14 implemented)

| Key | Name | Type | GPU | Reference |
|-----|------|------|-----|-----------|
| `wiener` | Wiener Filter | Classical | No |  |
| `traditional_cpu` | FISTA-TV | Iterative | No | Beck & Teboulle 2009 |
| `angular_spectrum` | Angular Spectrum Method | Classical | No | Goodman 2005 |
| `fresnel_backprop` | Fresnel Back-Propagation | Classical | No | Schnars & Jueptner 2005 |
| `tikhonov` | Tikhonov Regularisation | Classical | No |  |
| `admm_tv` | ADMM-TV | Iterative | No | Boyd et al. 2011 |
| `ista_l1` | ISTA-L1 | Iterative | No | Daubechies et al. 2004 |
| `residual_min` | Residual Minimisation | Iterative | No | PWM Scenario IV |
| `dl_hologan` | HoloGAN-CS | Deep Learning | Yes | Wu et al. 2020 |
| `dl_deepfresnel` | DeepFresnel | Deep Learning | Yes | Rivenson et al. 2021 |
| `dl_holonet_cs` | HoloNet-CS | Deep Learning | Yes | Wang et al. 2022 |
| `dl_transformer` | CompHolo-Transformer | Deep Learning | Yes | Li et al. 2023 |
| `best_quality` | Diffusion-Holo | Deep Learning | Yes | Zhang et al. 2024 |
| `famous_dl` | PnP-PGD (DRUNet) | Deep Learning | Yes | Hurault et al. 2022 |

### Algorithm Catalog (24 total)

#### Classical (5)
1. **Off-axis holography** -- Carrier-fringe demodulation for amplitude/phase separation
2. **Fresnel back-propagation** -- Numerical propagation using the Fresnel kernel
3. **Angular Spectrum Method** -- Exact diffraction propagation via transfer function
4. **Phase-shifting** -- Multi-exposure interferometric phase extraction
5. **Gabor holography** -- In-line holographic recording with twin-image artifact

#### Iterative (10)
1. **ISTA-L1** -- Iterative Shrinkage-Thresholding with L1 sparsity prior
2. **TwIST** -- Two-step IST for faster convergence
3. **FISTA-TV** -- Fast IST with total-variation regularisation
4. **ADMM-TV** -- Alternating Direction Method of Multipliers with TV
5. **Tikhonov** -- L2-regularised least-squares (ridge regression)
6. **Residual minimisation** -- Grid-search over propagation distance
7. **PnP-PGD** -- Plug-and-Play Proximal Gradient Descent
8. **Alternating projections** -- Gerchberg-Saxton type iterative phase retrieval
9. **GPSR** -- Gradient Projection for Sparse Reconstruction
10. **Bregman splitting** -- Split Bregman method for L1/TV problems

#### Deep Learning (9)
1. **HoloGAN-CS** -- GAN-based compressive holographic reconstruction (2020)
2. **DeepFresnel** -- Learned Fresnel propagation network (2021)
3. **HoloNet-CS** -- Compressive holographic neural network (2022)
4. **CompHolo-Transformer** -- Vision Transformer for holographic reconstruction (2023)
5. **Diffusion-Holo** -- Score-based diffusion model for holography (2024)
6. **PnP deep denoiser** -- Plug-and-Play with DRUNet backbone
7. **Self-supervised** -- Self-supervised holographic reconstruction
8. **PINN** -- Physics-Informed Neural Network for wave propagation
9. **Foundation model** -- Large-scale pretrained model adapted for holography

**Total: 24 algorithms** (5 Classical + 10 Iterative + 9 Deep Learning)

## Usage

```python
from algorithm_base.compressive_holography.solvers import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

from algorithm_base.compressive_holography.solvers import run_fista_tv
x_hat = run_fista_tv(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| Diffusion-Holo | 2024 | 34.5 | -- | pending |
| CompHolo-Transformer | 2023 | 34.0 | -- | pending |
| HoloNet-CS | 2022 | 32.5 | -- | pending |
| ADMM-TV | 2011 | 31.0 | -- | pending |
| FISTA-TV | 2009 | 30.5 | -- | pending |
| ISTA-L1 | 2004 | 28.0 | -- | pending |
| Fresnel back-propagation | 2005 | 22.0 | -- | pending |
| Wiener Filter | -- | 20.0 | -- | pending |
| Angular Spectrum Method | 2005 | 21.0 | -- | pending |
| Tikhonov | -- | 20.0 | -- | pending |
