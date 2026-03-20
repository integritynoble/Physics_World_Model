# Single-Pixel Camera (SPC)

**Input**: photon count frames (T × H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spc/public/`

## Algorithms (25 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | TVAL3 |  | CPU |
| `best_quality` | ADMM-L1 |  | CPU |
| `fista_l1` | FISTA-L1 |  | CPU |
| `omp` | OMP |  | CPU |
| `cosamp` | CoSaMP |  | CPU |
| `iht` | IHT |  | CPU |
| `gap_tv` | GAP-TV |  | CPU |
| `twist` | TwIST |  | CPU |
| `ist` | IST |  | CPU |
| `gpsr` | GPSR |  | CPU |
| `wiener` | Wiener Filter |  | CPU |
| `richardson_lucy` | Richardson-Lucy |  | CPU |
| `tikhonov` | Tikhonov Regularization |  | CPU |
| `bm3d_amp` | BM3D-AMP |  | CPU |
| `damp` | D-AMP |  | CPU |
| `famous_dl` | ISTA-Net+ |  | GPU |
| `small_gpu` | ReconNet |  | GPU |
| `ista_net_plus` | ISTA-Net+ v2 |  | GPU |
| `hatnet` | HATNet |  | GPU |
| `scsnet` | SCSNet |  | GPU |
| `csnet_plus` | CSNet+ |  | GPU |
| `opine_net` | OPINE-Net+ |  | GPU |
| `transcs` | TransCS |  | GPU |
| `csgm` | CSGM |  | GPU |
| `dpir_spc` | DPIR |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.spc.solvers import run_solver, list_solvers
list_solvers()                    # 25 algorithms
y = ...                           # photon count frames (T × H × W, uint16)
x = run_solver('best_quality', y) # swap key to compare
```

## Papers

- `papers/inversenet/SPC_IMPLEMENTATION_COMPLETE.md`
- `papers/inversenet/SPC_RESULTS.md`
