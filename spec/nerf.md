# Neural Radiance Fields (NeRF)

**Input**: posed images (N × H × W × 3, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nerf/public/`

## Algorithms (6 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | SfM + MVS |  | CPU |
| `famous_dl` | NeRF (original MLP) |  | CPU |
| `small_gpu` | Instant-NGP |  | CPU |
| `rl_proxy` | Richardson-Lucy (proxy baseline) |  | CPU |
| `fista_proxy` | FISTA-TV (proxy baseline) |  | CPU |
| `best_quality` | Mip-NeRF 360 |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.nerf.solvers import run_solver, list_solvers
list_solvers()                    # 6 algorithms
y = ...                           # posed images (N × H × W × 3, float32)
x = run_solver('best_quality', y) # swap key to compare
```
