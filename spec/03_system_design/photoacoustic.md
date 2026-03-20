# Photoacoustic Imaging — System Design

```
[Laser pulse] → [Tissue] → [Ultrasound array] → y
                                       ↓
             [TR / DAS / model-based] → x
                      ↓ speed-of-sound calibration
```

**Mismatch**: speed of sound `[1480, 1560] m/s`
**Input**: time-series (elements × time, float32)  **Algorithms**: 16 — see `spec/photoacoustic.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/photoacoustic/public/`

```python
from algorithm_base.photoacoustic.solvers import run_solver
from pwm_core.mismatch.operators import pa_calibrate_sos
c0 = pa_calibrate_sos(y)
calib_cfg = {"c0": float(c0)}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
