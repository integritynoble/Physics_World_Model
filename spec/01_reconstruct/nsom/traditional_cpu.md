# Near-field Scanning Optical Microscopy (NSOM) — Richardson-Lucy

**CPU**
**Input**: near-field signal (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/nsom/public/`

```python
from algorithm_base.nsom.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
