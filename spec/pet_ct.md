# PET/CT Fusion

**Input**: PET sinogram + CT projections (both float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pet_ct/public/`

## Algorithms (3 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | Adjoint [proxy] |  | CPU |
| `best_quality` | PnP-ADMM [proxy] |  | CPU |
| `petct_dl` | PET-CT-Fusion-Net [proxy] |  | CPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.pet_ct.solvers import run_solver, list_solvers
list_solvers()                    # 3 algorithms
y = ...                           # PET sinogram + CT projections (both float32)
x = run_solver('best_quality', y) # swap key to compare
```
