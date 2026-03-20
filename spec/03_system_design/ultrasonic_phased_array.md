# Ultrasonic Phased Array (TFM/FMC) — System Design

```
[Source] → [Forward (Ultrasonic Phased Array (TFM/FMC))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: FMC data (elem × elem × time, float32)  **Algorithms**: 15 — see `spec/ultrasonic_phased_array.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasonic_phased_array/public/`

```python
from algorithm_base.ultrasonic_phased_array.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
