# Brillouin Microscopy — Spec-CNN

**GPU**  *CNN for spectroscopy, 2018*
**Input**: spectral shift map (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/brillouin/public/`

```python
from algorithm_base.brillouin.solvers import run_solver
x = run_solver('dl_cnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
