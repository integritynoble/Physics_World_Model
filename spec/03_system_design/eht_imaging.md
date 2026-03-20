# Event Horizon Telescope (EHT) Imaging — System Design

```
[Source] → [Forward (Event Horizon Telescope (EHT) Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: hologram (H × W, complex64)  **Algorithms**: 15 — see `spec/eht_imaging.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/eht_imaging/public/`

```python
from algorithm_base.eht_imaging.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
