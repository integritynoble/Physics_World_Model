# Magnetic Resonance Imaging (MRI) — Proximal Gradient Descent

**CPU**  *Combettes & Wajs, Multiscale Model Simul 2005*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
x = run_solver('proximal_gradient', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
