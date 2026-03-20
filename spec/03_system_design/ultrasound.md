# Ultrasound B-mode Imaging — System Design

```
[Transducer array] → [Tissue] → [Echoes] → RF y
                                      ↓
                    [DAS / DMAS / TV] → x
                       ↓ speed-of-sound calibration
```

**Mismatch**: speed of sound `[1400, 1600] m/s`
**Input**: RF data (elements × samples, float32)  **Algorithms**: 17 — see `spec/ultrasound.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/public/`

```python
from algorithm_base.ultrasound.solvers import run_solver
from pwm_core.mismatch.operators import ultrasound_calibrate_sos
c0 = ultrasound_calibrate_sos(y)
calib_cfg = {"c0": float(c0)}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
