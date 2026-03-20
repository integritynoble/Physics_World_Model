# X-ray NDT (Radiography) — XR-Mamba

**GPU**  *SSM for X-ray, 2026*
**Input**: projection (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xray_ndt/public/`

```python
from algorithm_base.xray_ndt.solvers import run_solver
x = run_solver('dl_mamba', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
