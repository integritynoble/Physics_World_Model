# Magnetic Resonance Imaging (MRI) — MoDL (5 unrolls)

**GPU**  *Aggarwal et al., IEEE TMI 2019*
**Input**: k-space (H × W × 2: real+imag, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`

```python
from algorithm_base.mri.solvers import run_solver
cfg = {'n_iter': 5}
x = run_solver('small_gpu', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
