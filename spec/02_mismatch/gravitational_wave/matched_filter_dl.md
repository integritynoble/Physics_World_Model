# Gravitational Wave Detection — GW-DL (PyCBC-ML) [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: strain (samples, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/gravitational_wave/public/`

```python
from algorithm_base.gravitational_wave.solvers import run_solver


x_wrong = run_solver('matched_filter_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('matched_filter_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
