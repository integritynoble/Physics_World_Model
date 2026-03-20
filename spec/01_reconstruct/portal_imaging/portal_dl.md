# Portal Imaging (EPID) — PortalDL [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: EPID projection (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/portal_imaging/public/`

```python
from algorithm_base.portal_imaging.solvers import run_solver
x = run_solver('portal_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
