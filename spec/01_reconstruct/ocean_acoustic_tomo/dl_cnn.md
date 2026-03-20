# Ocean Acoustic Tomography — RS-CNN

**GPU**  *Deep learning for remote sensing, 2018*
**Input**: travel times (pairs, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ocean_acoustic_tomo/public/`

```python
from algorithm_base.ocean_acoustic_tomo.solvers import run_solver
x = run_solver('dl_cnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
