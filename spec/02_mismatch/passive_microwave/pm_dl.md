# Passive Microwave Radiometry — PM-Net [proxy] + Gradient

**CPU**  **Mismatch**: operator model error `modality-dependent`
**Input**: brightness T (H × W × ch, float32)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/passive_microwave/public/`

```python
from algorithm_base.passive_microwave.solvers import run_solver


x_wrong = run_solver('pm_dl', y)           # no correction
# auto-calibrate mismatch parameter
calib_cfg = {"mismatch_param": None}
x = run_solver('pm_dl', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
