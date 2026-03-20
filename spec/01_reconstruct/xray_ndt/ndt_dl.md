# X-ray NDT (Radiography) — NDT-DefectNet [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: projection (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xray_ndt/public/`

```python
from algorithm_base.xray_ndt.solvers import run_solver
x = run_solver('ndt_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
