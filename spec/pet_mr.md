# PET/MR Fusion

**Input**: PET sinogram + MRI k-space (both float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pet_mr/public/`

## Algorithms (3 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Adjoint [proxy] |  | CPU |
| `best_quality` | PnP-ADMM [proxy] |  | CPU |
| `petmr_dl` | PET-MR-DeepJoint [proxy] |  | CPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.pet_mr.solvers import run_solver, list_solvers
list_solvers()                    # 3 algorithms
y = ...                           # PET sinogram + MRI k-space (both float32)
x = run_solver('best_quality', y) # swap key to compare
```
