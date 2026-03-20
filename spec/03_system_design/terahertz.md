# Terahertz Imaging (THz) — System Design

```
[Source] → [Forward (Terahertz Imaging (THz))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: THz waveform (T × H × W, float32)  **Algorithms**: 15 — see `spec/terahertz.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/terahertz/public/`

```python
from algorithm_base.terahertz.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
