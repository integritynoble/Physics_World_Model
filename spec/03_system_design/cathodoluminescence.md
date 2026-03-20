# Cathodoluminescence (CL) Imaging — System Design

```
[Source] → [Forward (Cathodoluminescence (CL) Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: spectrum image (H × W × λ, float32)  **Algorithms**: 15 — see `spec/cathodoluminescence.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/cathodoluminescence/public/`

```python
from algorithm_base.cathodoluminescence.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
