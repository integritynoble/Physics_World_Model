# Correlative Light-Electron Microscopy (CLEM) — System Design

```
[Source] → [Forward (Correlative Light-Electron Microscopy (CLEM))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: EM + fluorescence (H × W, float32)  **Algorithms**: 15 — see `spec/clem.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/clem/public/`

```python
from algorithm_base.clem.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
