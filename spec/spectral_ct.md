# Photon-Counting Spectral CT

**Input**: energy-bin sinograms (bins × angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spectral_ct/public/`

## Algorithms (3 total)

| Key | Name | PSNR | Device |
|-----|------|------|--------|
| `traditional_cpu` | FBP [proxy] |  | CPU |
| `best_quality` | DL-Recon [proxy] |  | CPU |
| `spectral_ct_dl` | SpectralCT-Net [proxy] |  | CPU |

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
from algorithm_base.spectral_ct.solvers import run_solver, list_solvers
list_solvers()                    # 3 algorithms
y = ...                           # energy-bin sinograms (bins × angles × detectors, float32)
x = run_solver('best_quality', y) # swap key to compare
```
