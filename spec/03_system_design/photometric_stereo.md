# Photometric Stereo — System Design

```
[Source] → [Forward (Photometric Stereo)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: images under N lights (N × H × W, float32)  **Algorithms**: 15 — see `spec/photometric_stereo.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/photometric_stereo/public/`

```python
from algorithm_base.photometric_stereo.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
