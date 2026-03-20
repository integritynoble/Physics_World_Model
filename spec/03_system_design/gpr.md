# Ground-Penetrating Radar (GPR) — System Design

```
[Source] → [Forward (Ground-Penetrating Radar (GPR))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: B-scan (traces × samples, float32)  **Algorithms**: 15 — see `spec/gpr.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/gpr/public/`

```python
from algorithm_base.gpr.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
