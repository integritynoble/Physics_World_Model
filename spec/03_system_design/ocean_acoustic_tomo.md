# Ocean Acoustic Tomography — System Design

```
[Source] → [Forward (Ocean Acoustic Tomography)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: travel times (pairs, float32)  **Algorithms**: 15 — see `spec/ocean_acoustic_tomo.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ocean_acoustic_tomo/public/`

```python
from algorithm_base.ocean_acoustic_tomo.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
