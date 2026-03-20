# CT + Fluorescence (FLIT) — PnP-ADMM [proxy]

**CPU**  *Richardson 1972, JOSA*
**Input**: XRF sinogram (angles × detectors × ch, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ct_fluorescence/public/`

```python
from algorithm_base.ct_fluorescence.solvers import run_solver
x = run_solver('best_quality', y)
# PSNR/SSIM: from pwm_core.utils.metrics import compute_psnr, compute_ssim
```
