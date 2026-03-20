# Doppler Ultrasound — System Design

```
[Source] → [Forward (Doppler Ultrasound)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: IQ data (H × W × T, float32)  **Algorithms**: 15 — see `spec/doppler_ultrasound.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/doppler_ultrasound/public/`

```python
from algorithm_base.doppler_ultrasound.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
