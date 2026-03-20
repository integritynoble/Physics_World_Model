# Interferometric SAR (InSAR) — RS-CNN

**GPU**  *Deep learning for remote sensing, 2018*
**Input**: interferometric phase (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/insar/public/`

```python
from algorithm_base.insar.solvers import run_solver
x = run_solver('dl_cnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
