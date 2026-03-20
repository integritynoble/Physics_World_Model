# X-ray Computed Tomography (CT) — ART

**CPU**  *Gordon, Bender & Herman, J Theor Biol 1970*
**Input**: sinogram (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`

```python
from algorithm_base.ct.solvers import run_solver
cfg = {'iters': 5, 'relaxation': 0.1}
x = run_solver('art', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
