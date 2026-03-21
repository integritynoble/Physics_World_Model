# Single-Pixel Camera (SPC) (`spc`)

Category: Compressive Imaging

## Solvers (38 total: 22 classical + 16 deep learning)

### Classical / Iterative (22)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `traditional_cpu` | TVAL3 | No | Li et al., Rice CAAM Tech Report 2009 |
| `best_quality` | ADMM-L1 | No | Boyd et al., Found. Trends ML 2010 |
| `fista_l1` | FISTA-L1 | No | Beck & Teboulle, SIAM J. Imaging Sci. 2009 |
| `omp` | OMP | No | Pati, Rezaiifar & Krishnaprasad, Asilomar 1993 |
| `cosamp` | CoSaMP | No | Needell & Tropp, Appl. Comput. Harmon. Anal. 2009 |
| `iht` | IHT | No | Blumensath & Davies, J. Fourier Anal. Appl. 2009 |
| `gap_tv` | GAP-TV | No | Yuan, ICIP 2016 |
| `twist` | TwIST | No | Bioucas-Dias & Figueiredo, IEEE TIP 2007 |
| `ist` | IST | No | Daubechies et al., Comm. Pure Appl. Math 2004 |
| `gpsr` | GPSR | No | Figueiredo, Nowak & Wright, IEEE JSTSP 2007 |
| `wiener` | Wiener Filter | No | Wiener, MIT Press 1949 |
| `richardson_lucy` | Richardson-Lucy | No | Richardson, JOSA 1972; Lucy, Astron. J. 1974 |
| `tikhonov` | Tikhonov Regularization | No | Tikhonov 1963; Hansen, SIAM 1998 |
| `bm3d_amp` | BM3D-AMP | No | Metzler, Maleki & Baraniuk, IEEE TIT 2016 |
| `damp` | D-AMP | No | Metzler, Maleki & Baraniuk, ISIT 2014 |
| `basis_pursuit` | Basis Pursuit | No | Chen, Donoho & Saunders, SIAM Review 1998 |
| `subspace_pursuit` | Subspace Pursuit | No | Dai & Milenkovic, IEEE TIT 2009 |
| `sl0` | Smoothed L0 (SL0) | No | Mohimani, Babaie-Zadeh & Jutten, IEEE TSP 2009 |
| `amp` | AMP | No | Donoho, Maleki & Montanari, PNAS 2009 |
| `niht` | Normalized IHT | No | Blumensath, Sampling Theory in Signal & Image Proc. 2010 |
| `htp` | Hard Thresholding Pursuit | No | Foucart, Appl. Comput. Harmon. Anal. 2011 |
| `admm_tv` | ADMM-TV | No | Boyd et al., Found. Trends ML 2011 |

### Deep Learning (16)

| Key | Name | GPU | Reference |
|-----|------|-----|-----------|
| `famous_dl` | ISTA-Net+ | Yes | Zhang & Ghanem, CVPR 2018 |
| `small_gpu` | ReconNet | Yes | Kulkarni et al., CVPR 2016 |
| `ista_net_plus` | ISTA-Net+ v2 | Yes | Zhang & Ghanem, CVPR 2018 (DRS variant) |
| `hatnet` | HATNet | Yes | Song et al., IEEE TIP 2021 |
| `scsnet` | SCSNet | Yes | Shi et al., IEEE TCSVT 2019 |
| `csnet_plus` | CSNet+ | Yes | Shi et al., IEEE TIP 2020 |
| `opine_net` | OPINE-Net+ | Yes | Zhang et al., IEEE TCSVT 2020 |
| `transcs` | TransCS | Yes | Shen et al., IEEE TIP 2022 |
| `csgm` | CSGM | Yes | Bora et al., ICML 2017 |
| `dpir_spc` | DPIR | Yes | Zhang et al., IEEE TPAMI 2022 |
| `pnp_hqs_drunet` | PnP-HQS (DRUNet) | Yes | Zhang et al., CVPR 2017 |
| `amp_net` | AMP-Net | Yes | Zhang et al., IEEE TIP 2021 |
| `csformer` | CSFormer | Yes | Ye et al., NeurIPS 2023 |
| `diffcs` | DiffCS | Yes | Diffusion model for CS reconstruction, 2024 |
| `fsoinet` | FSOINet | Yes | Chen et al., CVPR 2023 |
| `spc_foundation` | SPC-Foundation | Yes | Foundation model for compressive sensing, 2025 |

### Algorithms Not Yet Implemented

The following algorithms from the literature are documented but not yet implemented as solvers:

- **LASSO** -- Least Absolute Shrinkage and Selection Operator (Tibshirani, J. Royal Statist. Soc. 1996)
- **Dantzig Selector** -- (Candes & Tao, Ann. Statist. 2007)
- **Model-based CS** -- structured sparsity (Baraniuk et al., IEEE TIT 2010)
- **TAMP** -- Turbo AMP (Schniter, Asilomar 2010)
- **VAMP** -- Vector AMP (Rangan et al., ISIT 2017)
- **LDAMP** -- Learned D-AMP (Metzler et al., NeurIPS 2017)
- **COAST** -- COntrollable Arbitrary-Sampling neTwork (You et al., IEEE TIP 2021)
- **DPC-DUN** -- Dual-Path CS Deep Unfolding (Song et al., CVPR 2023)
- **CASNet** -- Content-Aware Scalable CS (Chen et al., IEEE TIP 2022)

## Usage

```python
# Import and run
from algorithm_base.spc import run_solver
x_hat = run_solver("traditional_cpu", y, operator)

# Or use specific function
from algorithm_base.spc import run_traditional_cpu
x_hat = run_traditional_cpu(y, operator)
```

## Algorithm Leaderboard

| Algorithm | Year | Ref PSNR | PWM PSNR | Status |
|-----------|------|----------|----------|--------|
| AMP-Net | 2021 | 34.6 | 27.2 | partial |
| ISTA-Net+ | 2018 | 32.3 | 27.2 | partial |
| TransCS | 2022 | 31.1 | 27.2 | partial |
| CSNet+ | 2019 | 29.8 | 27.2 | done |
| TVAL3 | 2009 | 24.6 | 27.2 | done |
| Random sampling baseline | 2009 | 15.0 | 27.2 | done |
| Pseudoinverse (no regularization) | 2009 | 8.0 | 27.2 | done |
| ADMM-L1 (PWM) | — | 6.8 | 27.2 | done |
