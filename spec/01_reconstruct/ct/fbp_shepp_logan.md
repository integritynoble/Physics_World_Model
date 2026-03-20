# X-ray Computed Tomography (CT) — FBP (Shepp-Logan)

**CPU**  *Shepp & Logan, IEEE TNS 1974*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
cfg = {'filter': 'shepp_logan'}
x = run_solver('fbp_shepp_logan', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
