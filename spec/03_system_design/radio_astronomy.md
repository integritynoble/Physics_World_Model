# Radio Aperture Synthesis — System Design

```
[Source] → [Forward (Radio Aperture Synthesis)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: visibilities (baselines × freq × T, complex64)  **Algorithms**: 15 — see `spec/radio_astronomy.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/radio_astronomy/public/`

```python
from algorithm_base.radio_astronomy.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
