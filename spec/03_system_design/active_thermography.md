# Active Thermography (IR) — System Design

```
[Source] → [Forward (Active Thermography (IR))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: thermal sequence (T × H × W, float32)  **Algorithms**: 15 — see `spec/active_thermography.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/active_thermography/public/`

```python
from algorithm_base.active_thermography.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
