# Low-Dose Widefield Microscopy — System Design

```
[Source] → [Forward (Low-Dose Widefield Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: photon-limited image (H × W, float32)  **Algorithms**: 16 — see `spec/widefield_lowdose.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield_lowdose/public/`

```python
from algorithm_base.widefield_lowdose.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
