# Wide-Angle X-ray Scattering (WAXS) — prDeep

**GPU**  *Deep phase retrieval, 2020*
**Input**: wide-angle pattern (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/waxs/public/`

```python
from algorithm_base.waxs.solvers import run_solver
x = run_solver('dl_prdeep', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
