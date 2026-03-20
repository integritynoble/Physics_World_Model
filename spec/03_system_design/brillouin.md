# Brillouin Microscopy — System Design

```
[Source] → [Forward (Brillouin Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: spectral shift map (H × W, float32)  **Algorithms**: 15 — see `spec/brillouin.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/brillouin/public/`

```python
from algorithm_base.brillouin.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
