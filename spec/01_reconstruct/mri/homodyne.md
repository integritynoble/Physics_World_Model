# Magnetic Resonance Imaging (MRI) — Homodyne Detection

**CPU**  *Noll, Nishimura, Macovski, IEEE TMI 1991*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
x = run_solver('homodyne', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
