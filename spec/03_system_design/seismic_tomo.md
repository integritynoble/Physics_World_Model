# Seismic Tomography — System Design

```
[Source] → [Forward (Seismic Tomography)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: travel times (src-recv, float32)  **Algorithms**: 15 — see `spec/seismic_tomo.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/seismic_tomo/public/`

```python
from algorithm_base.seismic_tomo.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
