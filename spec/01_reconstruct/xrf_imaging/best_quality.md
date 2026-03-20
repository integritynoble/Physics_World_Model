# X-ray Fluorescence (XRF) Imaging — PnP-ADMM [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: fluorescence map (H × W × elements, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xrf_imaging/public/`

```python
from algorithm_base.xrf_imaging.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
