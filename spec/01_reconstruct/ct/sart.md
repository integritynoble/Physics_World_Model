# X-ray Computed Tomography (CT) — SART

**CPU**  **PSNR**: ~29.1 dB  *Andersen & Kak, Ultrason Imaging 1984 — 29.1 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
cfg = {'iters': 10, 'relaxation': 0.25}
x = run_solver('sart', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
