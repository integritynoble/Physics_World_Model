# Electron Backscatter Diffraction (EBSD) — System Design

```
[Source] → [Forward (Electron Backscatter Diffraction (EBSD))] → [Detector] → y
              ↓
          [Mismatch]
```

**Mismatch**: operator model error `modality-dependent`
**Input**: Kikuchi pattern (H × W × px × py, float32)  **Algorithms**: 15 — see `spec/ebsd.md`
**Benchmark**: `gs://pwm-benchmark-datasets/datasets/Benchmark/ebsd/public/`

```python
from algorithm_base.ebsd.solvers import run_solver


calib_cfg = {}
x = run_solver('traditional_cpu', y, cfg=calib_cfg)
```
