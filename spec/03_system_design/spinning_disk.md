# Spinning Disk Confocal Microscopy — System Design

```
[Source] → [Forward (Spinning Disk Confocal Microscopy)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: Z-stack (Z × H × W, float32)  **Algorithms**: 15 — see `spec/spinning_disk.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/spinning_disk/public/`

```python
from algorithm_base.spinning_disk.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
