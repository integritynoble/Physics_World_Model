# DESI Mass Spectrometry Imaging — Adjoint [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: mass image (H × W × m/z, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/desi/public/`

```python
from algorithm_base.desi.solvers import run_solver
x = run_solver('traditional_cpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
