# Weather / Doppler Radar — System Design

```
[Source] → [Forward (Weather / Doppler Radar)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: reflectivity (H × W, float32)  **Algorithms**: 15 — see `spec/weather_radar.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/weather_radar/public/`

```python
from algorithm_base.weather_radar.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
