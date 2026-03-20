# X-ray Radiography — System Design

```
[Source] → [Forward (X-ray Radiography)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: attenuation image (H × W, float32)  **Algorithms**: 15 — see `spec/xray_radiography.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/xray_radiography/public/`

```python
from algorithm_base.xray_radiography.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
