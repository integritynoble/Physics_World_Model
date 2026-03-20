# Phase Contrast Microscopy — System Design

```
[Source] → [Forward (Phase Contrast Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: image (H × W, float32)  **Algorithms**: 15 — see `spec/phase_contrast.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/phase_contrast/public/`

```python
from algorithm_base.phase_contrast.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
