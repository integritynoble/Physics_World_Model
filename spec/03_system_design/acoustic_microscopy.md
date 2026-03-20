# Scanning Acoustic Microscopy (SAM) — System Design

```
[Source] → [Forward (Scanning Acoustic Microscopy (SAM))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: RF data (H × W × T, float32)  **Algorithms**: 15 — see `spec/acoustic_microscopy.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/acoustic_microscopy/public/`

```python
from algorithm_base.acoustic_microscopy.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
