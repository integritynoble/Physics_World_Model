# X-ray Computed Tomography (CT) — SIRT

**CPU**  **PSNR**: ~29.5 dB  *Gilbert, J Theor Biol 1972 — 29.5 dB on LoDoPaB*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
cfg = {'iters': 20}
x = run_solver('sirt', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
