# Proton Radiography — System Design

```
[Source] → [Forward (Proton Radiography)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: fluence map (H × W, float32)  **Algorithms**: 15 — see `spec/proton_radiography.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/proton_radiography/public/`

```python
from algorithm_base.proton_radiography.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
