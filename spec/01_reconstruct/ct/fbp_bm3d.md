# X-ray Computed Tomography (CT) — FBP + BM3D

**CPU**  *Dabov et al., TIP 2007*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('fbp_bm3d', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
