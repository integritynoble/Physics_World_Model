# Scanning Tunneling Microscopy (STM) — Probe-CNN

**GPU**  *CNN for scanning probe, 2019*
**Input**: tunneling map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/stm/public/`

```python
from algorithm_base.stm.solvers import run_solver
x = run_solver('dl_cnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
