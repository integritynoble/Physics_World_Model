# Stellar Coronagraphy — System Design

```
[Source] → [Forward (Stellar Coronagraphy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: image (H × W, float32)  **Algorithms**: 15 — see `spec/coronagraphy.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/coronagraphy/public/`

```python
from algorithm_base.coronagraphy.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
