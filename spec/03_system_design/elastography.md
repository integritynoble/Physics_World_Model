# Shear-Wave Elastography — System Design

```
[Source] → [Forward (Shear-Wave Elastography)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: displacement (H × W × 3, float32)  **Algorithms**: 15 — see `spec/elastography.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/elastography/public/`

```python
from algorithm_base.elastography.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
