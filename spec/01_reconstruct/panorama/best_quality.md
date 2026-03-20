# Panorama Multi-Focus Fusion — Guided Filter Fusion

**CPU**
**Input**: images (N × H × W × 3, uint8)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/panorama/public/`

```python
from algorithm_base.panorama.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
