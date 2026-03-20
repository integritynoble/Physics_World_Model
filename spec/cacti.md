# Coded Aperture Compressive Temporal Imaging (CACTI)

**Input**: coded frames (T/B × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cacti/public/`

## Algorithms (7 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | GAP-TV |  | CPU |
| `best_quality` | EfficientSCI |  | CPU |
| `famous_dl` | ELP-Unfolding |  | CPU |
| `small_gpu` | EfficientSCI-T |  | CPU |
| `pnp_ffdnet` | PnP-FFDNet |  | CPU |
| `hisvit9` | HiSViT-9 |  | GPU |
| `hisvit13` | HiSViT-13 |  | GPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.cacti.solvers import run_solver, list_solvers
list_solvers()                    # 7 algorithms
y = ...                           # coded frames (T/B × H × W, float32)
x = run_solver('best_quality', y) # swap key to compare
```

## Papers

- `papers/inversenet/CACTI_IMPLEMENTATION_COMPLETE.md`
