# X-ray Computed Tomography (CT) — FBP + TV

**CPU**  *Rudin, Osher & Fatemi, Physica D 1992*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
x = run_solver('fbp_tv', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
