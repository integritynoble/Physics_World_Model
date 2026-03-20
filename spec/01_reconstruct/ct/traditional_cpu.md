# X-ray Computed Tomography (CT) — FBP (Ram-Lak)

**CPU**  *Ramachandran & Lakshminarayanan 1971*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
cfg = {'filter': 'ramlak'}
x = run_solver('traditional_cpu', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
