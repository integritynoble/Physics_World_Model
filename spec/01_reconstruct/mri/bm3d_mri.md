# Magnetic Resonance Imaging (MRI) — BM3D-MRI

**CPU**  *Eksioglu, IEEE SPL 2016*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
x = run_solver('bm3d_mri', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
