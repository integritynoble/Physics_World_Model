# Light-Sheet Fluorescence Microscopy (LSFM) — System Design

```
[Source] → [Forward (Light-Sheet Fluorescence Microscopy (LSFM))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: Z-stack (Z × H × W, float32)  **Algorithms**: 16 — see `spec/lightsheet.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/lightsheet/public/`

```python
from algorithm_base.lightsheet.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
