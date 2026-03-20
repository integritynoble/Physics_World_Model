# Wide-Angle X-ray Scattering (WAXS) — PhaseNet + Gradient

**GPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: wide-angle pattern (H × W, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/waxs/public/`

```python
from algorithm_base.waxs.solvers import run_solver


x_wrong = run_solver('dl_phasenet', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('dl_phasenet', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
