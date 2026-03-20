# Shearography — System Design

```
[Source] → [Forward (Shearography)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: shearograms (2 × H × W, float32)  **Algorithms**: 15 — see `spec/shearography.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/shearography/public/`

```python
from algorithm_base.shearography.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
