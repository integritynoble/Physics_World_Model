# X-ray Computed Tomography (CT) — CGLS

**CPU**  **PSNR**: ~30.2 dB  *Hestenes & Stiefel 1952 — 30.2 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
cfg = {'iters': 15}
x = run_solver('cgls', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
