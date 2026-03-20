# Low-Dose Widefield Microscopy — CARE

**GPU**
**Input**: photon-limited image (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield_lowdose/public/`

```python
from algorithm_base.widefield_lowdose.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
