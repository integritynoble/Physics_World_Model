# Magnetic Resonance Imaging (MRI) — Tikhonov Regularization

**CPU**  *Tikhonov, Soviet Math Dokl 1963*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
x = run_solver('tikhonov', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
