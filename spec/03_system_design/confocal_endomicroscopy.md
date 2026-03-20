# Confocal Laser Endomicroscopy (CLE) — System Design

```
[Source] → [Forward (Confocal Laser Endomicroscopy (CLE))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: confocal frame (H × W, float32)  **Algorithms**: 15 — see `spec/confocal_endomicroscopy.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/confocal_endomicroscopy/public/`

```python
from algorithm_base.confocal_endomicroscopy.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
