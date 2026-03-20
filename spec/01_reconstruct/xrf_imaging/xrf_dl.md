# X-ray Fluorescence (XRF) Imaging — XRF-Net [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: fluorescence map (H × W × elements, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xrf_imaging/public/`

```python
from algorithm_base.xrf_imaging.solvers import run_solver
x = run_solver('xrf_dl', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
