# X-ray Computed Tomography (CT)

**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

## Algorithms (41 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FBP (Ram-Lak) |  | CPU |
| `fbp_shepp_logan` | FBP (Shepp-Logan) |  | CPU |
| `fbp_cosine` | FBP (Cosine) |  | CPU |
| `fbp_hamming` | FBP (Hamming) |  | CPU |
| `fbp_hann` | FBP (Hann) |  | CPU |
| `landweber` | Landweber |  | CPU |
| `art` | ART |  | CPU |
| `sirt` | SIRT | ~29.5 dB | CPU |
| `cgls` | CGLS | ~30.2 dB | CPU |
| `mlem` | MLEM |  | CPU |
| `sart` | SART | ~29.1 dB | CPU |
| `osem` | OSEM |  | CPU |
| `tikhonov` | Tikhonov |  | CPU |
| `tv_admm` | TV-ADMM | ~27.8 dB | CPU |
| `chambolle_pock` | Chambolle-Pock |  | CPU |
| `pnp_admm_nlm` | PnP-ADMM (NLM) | ~39.5 dB | CPU |
| `pnp_hqs_nlm` | PnP-HQS (NLM) | ~39.1 dB | CPU |
| `pnp_fista_nlm` | PnP-FISTA (NLM) |  | CPU |
| `pnp_admm_bm3d` | PnP-ADMM (BM3D) |  | CPU |
| `best_quality` | FBP + NLM | ~28.5 dB | CPU |
| `fbp_bm3d` | FBP + BM3D |  | CPU |
| `fbp_bilateral` | FBP + Bilateral |  | CPU |
| `fbp_wavelet` | FBP + Wavelet |  | CPU |
| `fbp_tv` | FBP + TV |  | CPU |
| `famous_dl` | RED-CNN | ~33.2 dB | CPU |
| `small_gpu` | RED-CNN |  | CPU |
| `fbpconvnet` | FBPConvNet | ~38.5 dB | GPU |
| `wgan_vgg` | WGAN-VGG | ~34.1 dB | GPU |
| `learn` | LEARN | ~43.1 dB | GPU |
| `learned_pd` | Learned Primal-Dual | ~36.2 dB | GPU |
| `iradonmap` | iRadonMAP | ~36.9 dB | GPU |
| `fbp_unet` | FBP + U-Net | ~35.8 dB | GPU |
| `dudonet` | DuDoNet | ~40.2 dB | GPU |
| `indudonet` | InDuDoNet | ~43.5 dB | GPU |
| `dudotrans` | DuDoTrans | ~42.1 dB | GPU |
| `ctformer` | CTformer | ~40.8 dB | GPU |
| `score_ct` | Score-CT | ~43.0 dB | GPU |
| `dps` | DPS | ~43.2 dB | GPU |
| `diffusion_mbir` | DiffusionMBIR | ~43.8 dB | GPU |
| `dolce` | DOLCE | ~36.0 dB | GPU |
| `ct_fm` | CT-FM | ~44.1 dB | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.ct.solvers import run_solver, list_solvers
list_solvers()                    # 41 algorithms
y = ...                           # sinogram (angles × detectors, float32)
x = run_solver('best_quality', y) # swap key to compare
```

## Mismatch

Key mismatch: **center-of-rotation offset (px)**  
Correction: `run_solver('best_quality', y, cfg={'calibrate': True})`

## Papers

- `papers/system_design/outputs/ct_forward_v1_iter1.md`
- `papers/system_design/outputs/ct_reconstruction_v1_iter1.md`
