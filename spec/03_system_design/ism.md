# Image Scanning Microscopy (ISM) — System Design

```
[Source] → [Forward (Image Scanning Microscopy (ISM))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: raw stack (H_scan × W_scan × px × py, float32)  **Algorithms**: 15 — see `spec/ism.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ism/public/`

```python
from algorithm_base.ism.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
