# Radio Interferometry (VLBI) — System Design

```
[Source] → [Forward (Radio Interferometry (VLBI))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: UV-plane data (N_baselines, complex64)  **Algorithms**: 15 — see `spec/radio_interferometry.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/radio_interferometry/public/`

```python
from algorithm_base.radio_interferometry.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
