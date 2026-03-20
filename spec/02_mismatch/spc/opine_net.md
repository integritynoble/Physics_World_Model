# Single-Pixel Camera (SPC) — OPINE-Net+ + Gradient

**GPU**  **Mismatch**: detector gain/bias `gain [0.8, 1.2], bias [-0.1, 0.1]`
**Input**: photon counts (T × H × W, uint16)
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spc/public/`

```python
from algorithm_base.spc.solvers import run_solver
from pwm_core.mismatch.operators import spc_calibrate_gain_bias

x_wrong = run_solver('opine_net', y)           # no correction
gain, bias = spc_calibrate_gain_bias(y)
calib_cfg = {"gain": float(gain), "bias": float(bias)}
x = run_solver('opine_net', y, cfg=calib_cfg)        # corrected
# compare: compute_psnr(x_true, x_wrong) vs compute_psnr(x_true, x)
```
