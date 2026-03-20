# Magnetic Resonance Imaging (MRI) — RED (Regularization by Denoising)

**CPU**  *Romano, Elad, Milanfar, SIAM J Imaging Sci 2017*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
x = run_solver('red_mri', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
