# Ptychographic Imaging (`ptychography`)

Category: Coherent Imaging / Phase Retrieval
Carrier: Electron. DAG: M→P→D.

## Solvers (25 total: 14 classical + 11 deep learning)

### Classical / Iterative (14)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `error_reduction` | Error Reduction | No | Fienup 1972 |
| `wdd` | Wigner Distribution Deconvolution (WDD) | No | Rodenburg & Bates 1992 |
| `difference_map` | Difference Map | No | Elser 2003 |
| `pie` | Ptychographic Iterative Engine (PIE) | No | Rodenburg & Faulkner 2004 |
| `raar` | Relaxed Averaged Alternating Reflections (RAAR) | No | Luke 2005 |
| `traditional_cpu` | Extended PIE (ePIE) | No | Maiden & Rodenburg 2009 |
| `mpie` | Momentum PIE (mPIE) | No | Maiden et al. 2012 |
| `landweber` | Landweber Iteration | No | Landweber 1951 |
| `tikhonov` | Tikhonov Regularization | No | Tikhonov 1963 |
| `tv_admm` | TV-ADMM | No | Boyd et al. 2011 |
| `pnp_admm_nlm` | PnP-ADMM with NLM | No | Venkatakrishnan et al. 2013 |
| `fpm` | Fourier Ptychography (FPM) | No | Zheng et al. 2013 |
| `sharp` | SHARP | No | Marchesini et al. 2013 |
| `amplitude_flow` | Amplitude Flow | No | Wang et al. 2017 |

### Deep Learning (11)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `best_quality` | PtychoNN (DL-PGD) | Yes | Cherukara et al. 2020 |
| `famous_dl` | AutoPhase (DL-PGD) | Yes | Nguyen et al. 2018 |
| `small_gpu` | PtychoNN 2.0 (DnCNN) | Yes | Wu et al. 2022 |
| `ptycho_diffusion` | Ptychography Diffusion (DL-PGD) | Yes | Cherukara et al. 2023 |
| `ptycho_former` | PtychoFormer (DL-DRS) | Yes | Shi et al. 2024 |
| `ptycho_mamba` | PtychoMamba (RED-DRUNet) | Yes | Li et al. 2024 |
| `pnp_pgd_drunet` | PnP-PGD DRUNet | Yes | Zhang et al. 2017 |
| `physics_nn` | PhysicsNN (DL-HQS) | Yes | Kellman et al. 2020 |
| `ptycho_dv` | PtychoDV (DL-DRS) | Yes | Zhou & Horstmeyer 2022 |
| `ptycho_flow` | PtychoFlow (DL-PGD) | Yes | Chang et al. 2023 |
| `ptycho_foundation` | PtychoFoundation (RED-DRUNet) | Yes | Zhang et al. 2025 |

### Algorithms Not Yet Implemented

The following algorithms from the literature are documented but not yet implemented as solvers:

- **rPIE** — regularized PIE (Maiden et al., Ultramicroscopy 2017)
- **OSS** — Oversampling Smoothness (Rodriguez et al., J. Appl. Crystallogr. 2013)
- **ML-Ptychography** — Maximum Likelihood (Thibault & Guizar-Sicairos, NJP 2012)
- **LSQ-ML** — Least-Squares ML (Odstrčil et al., Opt. Express 2018)
- **Position correction** — joint position + object (Maiden et al. 2012)
- **Multi-slice** — 3D thick specimen (Maiden et al., JOSA A 2012)
- **Blind ptychography** — joint probe + object (Thibault et al. 2009)
- **Mixed-state** — partial coherence (Thibault & Menzel, Nature 2013)
- **Wirtinger Flow** — for ptychography (Xu et al. 2018)
- **AD ptychography** — automatic differentiation (2018)
- **PtychoNet** — U-Net phase retrieval (2020)
- **Deep Ptychography** — physics-informed DL (Guzzi et al. 2022)
- **PtychoPINN** — Physics-Informed NN (2023)
- **Self-supervised ptychography** (2022)
- **4D-STEM DL** — deep learning reconstruction (Jiang et al. 2022)

## Usage

```python
# Import and run
from algorithm_base.ptychography import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.ptychography import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| PtychoFoundation | 2025 | 36.0 | 17.6 | gap |
| PtychoFormer | 2024 | 35.0 | 17.6 | gap |
| PtychoDV | 2022 | 34.0 | 17.6 | gap |
| AutoPhaseNN | 2022 | 33.0 | 17.6 | gap |
| ML-Ptychography | 2012 | 32.0 | — | not_impl |
| PtychoNN | 2020 | 31.0 | 17.6 | gap |
| ePIE | 2009 | 30.0 | 17.6 | gap |
| WDD | 1992 | 25.0 | 17.6 | gap |
| PIE | 2004 | 22.0 | 17.6 | partial |
| PtychoNN 2.0 (PWM) | 2022 | 21.0 | 17.6 | partial |

**Note:** The large gap between reference and PWM PSNR is due to the current forward model using Gaussian PSF convolution instead of the proper ptychographic model `y_j = |F{P(r-r_j) * O(r)}|^2`. This is a known issue requiring a physics model update.
