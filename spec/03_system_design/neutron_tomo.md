# Neutron Radiography / Tomography — System Design

```
[Source] → [Forward (Neutron Radiography / Tomography)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: projections (angles × H × W, float32)  **Algorithms**: 15 — see `spec/neutron_tomo.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/neutron_tomo/public/`

```python
from algorithm_base.neutron_tomo.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
