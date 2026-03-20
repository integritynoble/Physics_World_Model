# MR Angiography (MRA) — MRA-VesselNet [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: k-space (kx × ky × kz, complex64)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/mra/public/`

```python
from algorithm_base.mra.solvers import run_solver


x_wrong = run_solver('mra_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('mra_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
