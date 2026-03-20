# X-ray Computed Tomography (CT) — FBP (Cosine)

**CPU**  *Standard windowed FBP*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
cfg = {'filter': 'cosine'}
x = run_solver('fbp_cosine', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
