# Coded Aperture Snapshot Spectral Imaging (CASSI)

**Input**: coded snapshot (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/public/`

## Algorithms (22 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | GAP-TV | ~24.34 dB | CPU |
| `best_quality` | GAP-TV (200 iter) | ~24.9 dB | CPU |
| `small_gpu` | GAP-TV (fast) |  | CPU |
| `twist` | TwIST | ~23.1 dB | CPU |
| `famous_dl` | MST-L | ~34.81 dB | GPU |
| `mst_l` | MST-L | ~34.81 dB | GPU |
| `hdnet` | HDNet | ~34.66 dB | GPU |
| `hsi_sdecnn` | PnP-HSICNN | ~25.12 dB | GPU |
| `dauhst_9stg` | DAUHST-9stg | ~38.4 dB | GPU |
| `cst_l_plus` | CST-L-Plus | ~36.1 dB | GPU |
| `mst_plus_plus` | MST++ | ~36.0 dB | GPU |
| `dgsmp` | DGSMP | ~32.6 dB | GPU |
| `tsa_net` | TSA-Net | ~31.5 dB | GPU |
| `lambda_net` | λ-Net | ~30.1 dB | GPU |
| `admm_net` | ADMM-Net | ~29.1 dB | GPU |
| `gap_net` | GAP-Net | ~29.1 dB | GPU |
| `birnat` | BIRNAT |  | GPU |
| `bisrnet` | BiSRNet |  | GPU |
| `rdluf_mixs2_9stg` | RDLUF-MixS2-9stg | ~39.6 dB | GPU |
| `ssr_l` | SSR-L | ~34.0 dB | GPU |
| `padut_3stg` | PADUT-3stg | ~36.95 dB | GPU |
| `mijun_5stg` | MiJUN-5stg | ~40.9 dB | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.cassi.solvers import run_solver, list_solvers
list_solvers()                    # 22 algorithms
y = ...                           # coded snapshot (H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```

## Mismatch

Key mismatch: **dispersion step (px)**  
Correction: `run_solver('best_quality', y, cfg={'calibrate': True})`

## Papers

- `papers/system_design/outputs/spectral_lensless_forward_v1_iter1.md`
- `papers/inversenet/README_CASSI.md`
