# Coded Aperture Compressive Temporal Imaging (CACTI) — EfficientSCI

**CPU**  *Wang et al. CVPR 2023*
**Input**: coded frames (B × H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cacti/public/`

```python
from algorithm_base.cacti.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
