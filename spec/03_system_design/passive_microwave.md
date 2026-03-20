# Passive Microwave Radiometry — System Design

```
[Source] → [Forward (Passive Microwave Radiometry)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: brightness T (H × W × ch, float32)  **Algorithms**: 15 — see `spec/passive_microwave.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/passive_microwave/public/`

```python
from algorithm_base.passive_microwave.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
