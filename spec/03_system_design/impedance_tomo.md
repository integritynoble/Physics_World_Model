# Electrical Impedance Tomography (EIT) — System Design

```
[Source] → [Forward (Electrical Impedance Tomography (EIT))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: boundary voltages (M, float32)  **Algorithms**: 15 — see `spec/impedance_tomo.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/impedance_tomo/public/`

```python
from algorithm_base.impedance_tomo.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
