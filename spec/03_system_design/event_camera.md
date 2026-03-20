# Event Camera / Dynamic Vision Sensor (DVS) — System Design

```
[Source] → [Forward (Event Camera / Dynamic Vision Sensor (DVS))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: event stream (N × 4: t,x,y,p)  **Algorithms**: 15 — see `spec/event_camera.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/event_camera/public/`

```python
from algorithm_base.event_camera.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
