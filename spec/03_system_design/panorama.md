# Panorama Multi-Focus Fusion — System Design

```
[Source] → [Forward (Panorama Multi-Focus Fusion)] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: images (N × H × W × 3, uint8)  **Algorithms**: 16 — see `spec/panorama.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/panorama/public/`

```python
from algorithm_base.panorama.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
