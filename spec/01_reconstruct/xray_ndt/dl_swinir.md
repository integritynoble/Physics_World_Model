# X-ray NDT (Radiography) — XR-SwinIR

**GPU**  *SwinIR for X-ray, 2022*
**Input**: projection (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xray_ndt/public/`

```python
from algorithm_base.xray_ndt.solvers import run_solver
x = run_solver('dl_swinir', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
