# X-ray Computed Tomography (CT) — OSEM

**CPU**  *Hudson & Larkin, IEEE TMI 1994*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
cfg = {'iters': 5, 'n_subsets': 10}
x = run_solver('osem', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
