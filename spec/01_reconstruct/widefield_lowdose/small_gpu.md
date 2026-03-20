# Low-Dose Widefield Microscopy — Noise2Void

**CPU**
**Input**: photon-limited image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield_lowdose/public/`

```python
from algorithm_base.widefield_lowdose.solvers import run_solver
x = run_solver('small_gpu', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
