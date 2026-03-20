# Interferometric SAR (InSAR) — System Design

```
[Source] → [Forward (Interferometric SAR (InSAR))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: interferometric phase (H × W, float32)  **Algorithms**: 15 — see `spec/insar.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/insar/public/`

```python
from algorithm_base.insar.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
