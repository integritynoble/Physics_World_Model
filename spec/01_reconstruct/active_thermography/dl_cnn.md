# Active Thermography (IR) — Probe-CNN

**GPU**  *CNN for scanning probe, 2019*
**Input**: thermal sequence (T × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/active_thermography/public/`

```python
from algorithm_base.active_thermography.solvers import run_solver
x = run_solver('dl_cnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
