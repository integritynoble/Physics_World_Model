# Coded Exposure / Flutter Shutter — System Design

```
[Source] → [Forward (Coded Exposure / Flutter Shutter)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: coded frames (N × H × W, float32)  **Algorithms**: 15 — see `spec/coded_exposure.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/coded_exposure/public/`

```python
from algorithm_base.coded_exposure.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
