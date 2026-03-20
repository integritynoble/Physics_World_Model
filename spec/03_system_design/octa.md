# OCT Angiography (OCTA) — System Design

```
[Source] → [Forward (OCT Angiography (OCTA))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: B-scans (T × depth × A-scans, float32)  **Algorithms**: 15 — see `spec/octa.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/octa/public/`

```python
from algorithm_base.octa.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
