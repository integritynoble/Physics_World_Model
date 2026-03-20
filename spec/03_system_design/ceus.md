# Contrast-Enhanced Ultrasound (CEUS) — System Design

```
[Source] → [Forward (Contrast-Enhanced Ultrasound (CEUS))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: contrast frames (T × H × W, float32)  **Algorithms**: 15 — see `spec/ceus.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ceus/public/`

```python
from algorithm_base.ceus.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
