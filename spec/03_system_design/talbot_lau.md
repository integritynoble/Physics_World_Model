# Talbot-Lau X-ray Grating Interferometry — System Design

```
[Source] → [Forward (Talbot-Lau X-ray Grating Interferometry)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: stepping images (N_steps × H × W, float32)  **Algorithms**: 15 — see `spec/talbot_lau.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/talbot_lau/public/`

```python
from algorithm_base.talbot_lau.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
