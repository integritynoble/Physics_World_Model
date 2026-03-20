# Single Photon Emission CT (SPECT) — Richardson-Lucy

**CPU**  *Richardson 1972; Lucy 1974*
**Input**: projections (angles × detectors, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spect/public/`

```python
from algorithm_base.spect.solvers import run_solver
cfg = {'iters': 50}
x = run_solver('richardson_lucy', y, cfg=cfg)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
