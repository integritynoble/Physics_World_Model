# X-ray Computed Tomography (CT) — FBP + Bilateral

**CPU**  *Tomasi & Manduchi, ICCV 1998*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('fbp_bilateral', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
