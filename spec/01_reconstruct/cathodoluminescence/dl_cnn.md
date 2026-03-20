# Cathodoluminescence (CL) Imaging — Spec-CNN

**GPU**  *CNN for spectroscopy, 2018*
**Input**: spectrum image (H × W × λ, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cathodoluminescence/public/`

```python
from algorithm_base.cathodoluminescence.solvers import run_solver
x = run_solver('dl_cnn', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
