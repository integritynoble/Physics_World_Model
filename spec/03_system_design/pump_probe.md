# Pump-Probe Microscopy — System Design

```
[Source] → [Forward (Pump-Probe Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: transient spectra (T × λ, float32)  **Algorithms**: 15 — see `spec/pump_probe.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/pump_probe/public/`

```python
from algorithm_base.pump_probe.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
