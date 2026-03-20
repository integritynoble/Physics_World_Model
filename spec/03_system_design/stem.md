# Scanning Transmission Electron Microscopy (STEM) — System Design

```
[Source] → [Forward (Scanning Transmission Electron Microscopy (STEM))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: HAADF image (H × W, float32)  **Algorithms**: 15 — see `spec/stem.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/stem/public/`

```python
from algorithm_base.stem.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
