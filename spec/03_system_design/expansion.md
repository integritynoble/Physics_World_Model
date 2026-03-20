# Expansion Microscopy (ExM) — System Design

```
[Source] → [Forward (Expansion Microscopy (ExM))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: confocal + expansion (H × W, float32)  **Algorithms**: 15 — see `spec/expansion.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/expansion/public/`

```python
from algorithm_base.expansion.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
