# Differential Interference Contrast (DIC) — System Design

```
[Source] → [Forward (Differential Interference Contrast (DIC))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: DIC image (H × W, float32)  **Algorithms**: 15 — see `spec/dic.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/dic/public/`

```python
from algorithm_base.dic.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
