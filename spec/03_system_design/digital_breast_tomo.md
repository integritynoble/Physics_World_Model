# Digital Breast Tomosynthesis (DBT) — System Design

```
[Source] → [Forward (Digital Breast Tomosynthesis (DBT))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: projections (angles × H × W, float32)  **Algorithms**: 15 — see `spec/digital_breast_tomo.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/digital_breast_tomo/public/`

```python
from algorithm_base.digital_breast_tomo.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
