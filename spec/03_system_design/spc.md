# Single-Pixel Camera (SPC) — System Design

```
[Source] → [Forward (Single-Pixel Camera (SPC))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: detector gain/bias `gain [0.8, 1.2], bias [-0.1, 0.1]`
**Input**: photon counts (T × H × W, uint16)  **Algorithms**: 25 — see `spec/spc.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spc/public/`

```python
from algorithm_base.spc.solvers import run_solver
from pwm_core.mismatch.operators import spc_calibrate_gain_bias
gain, bias = spc_calibrate_gain_bias(y)
calib_cfg = {"gain": float(gain), "bias": float(bias)}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
