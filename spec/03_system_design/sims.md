# Secondary Ion Mass Spectrometry (SIMS) Imaging — System Design

```
[Source] → [Forward (Secondary Ion Mass Spectrometry (SIMS) Imaging)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: ion images (H × W × m/z, float32)  **Algorithms**: 15 — see `spec/sims.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/sims/public/`

```python
from algorithm_base.sims.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
