# Widefield Fluorescence Microscopy — System Design

```
[Source] → [Forward (Widefield Fluorescence Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: fluorescence image (H × W, float32)  **Algorithms**: 17 — see `spec/widefield.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/public/`

```python
from algorithm_base.widefield.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
